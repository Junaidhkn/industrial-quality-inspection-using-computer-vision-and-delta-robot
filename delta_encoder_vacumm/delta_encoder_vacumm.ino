#include <AccelStepper.h>
#include <math.h>

#define SOLENOID_PIN 16
#define ENCODER_A 18
#define ENCODER_B 19
#define ENCODER_Z 26

const long  PULSES_PER_REVOLUTION = 1000L;
const float DRIVE_ROLLER_DIAMETER = 0.065f;
const unsigned long SPEED_UPDATE_MS = 125;
const unsigned long TELEMETRY_MS    = 150;
const float ALPHA = 0.2f;
const bool ENABLE_SERIAL_PRINTS = false;
volatile long pulseCount = 0;
volatile long pulsesSinceLastZ = 0;
volatile long totalRevolutions = 0;
volatile unsigned long lastZMicros = 0;
volatile unsigned long revolutionTimeMicros = 0;
volatile bool zSeen = false;
volatile long totalPulses = 0;
float rpmAB = 0.0f;
float rpmZ = 0.0f;
float filteredRPM = 0.0f;
float linearSpeed_m_s = 0.0f;
float linearSpeed_mm_s = 0.0f;
#define MOTOR_COUNT 3
const int STEP_PINS[MOTOR_COUNT] = {2, 5, 8};
const int DIR_PINS[MOTOR_COUNT]  = {3, 6, 9};
const int ENA_PINS[MOTOR_COUNT]  = {4, 7, 10};
const float TARGET_RPM   = 400.0f;   // motor rpm
const float ACCEL_RPM    = 350.0f;   // motor rpm
const int MICROSTEPS   = 1600;
const float PULLEY_RATIO = 3.0f;
const int MOTOR_SIGN[MOTOR_COUNT] = { -1, -1, -1 };

AccelStepper motors[MOTOR_COUNT] = {
  AccelStepper(AccelStepper::DRIVER, STEP_PINS[0], DIR_PINS[0]),
  AccelStepper(AccelStepper::DRIVER, STEP_PINS[1], DIR_PINS[1]),
  AccelStepper(AccelStepper::DRIVER, STEP_PINS[2], DIR_PINS[2])
};


long jointAngleToMotorSteps(int motorIndex, float jointAngleDeg) {
  const float stepsPerJointDeg = ((float)MICROSTEPS * PULLEY_RATIO) / 360.0f;
  long steps = lround(jointAngleDeg * stepsPerJointDeg);
  return MOTOR_SIGN[motorIndex] * steps;
}

char incomingBuf[128];
int incomingPos = 0;
String activeMoveId = "";
bool home_reached = false;
float baseMaxStepPerSec = 0.0f;
float baseAcceleration  = 0.0f;
const float ACC_MULT_LIMIT = 0.6f;
const unsigned long SETTLE_MS = 80UL;
unsigned long lastTelemetryMs = 0;
unsigned long lastSpeedUpdateMs = 0;
unsigned long settleStartMs = 0;
bool settling = false;
bool telemetryEnabled = false;


String extractTag(const String &cmd) {
  int idx = cmd.lastIndexOf('@');
  if (idx < 0) return "";
  String tag = cmd.substring(idx);
  tag.trim();
  if (tag.length() >= 2) return tag;
  return "";
}

void printAckWithTag(const char* ack, const String &tag, int repeatCount = 2) {
  for (int i = 0; i < repeatCount; i++) {
    Serial.print(ack);
    if (tag.length() > 0) {
      Serial.print(" ");
      Serial.print(tag);
    }
    Serial.println();
    delay(2);
  }
}

void encoderA_ISR();
void encoderZ_ISR();
void calculateSpeed(unsigned long elapsedMs);
void publishTelemetry();
void processSerialInput();
void handleCommand(const String &cmd);
void startMoveAngles(const String &id, float a0, float a1, float a2, float speedMult);
bool motorsBusy();
void setSolenoid(bool on, const String &tag);
void stopAll();

void setup() {
  Serial.begin(115200);
  delay(50);
  Serial.println("FW_DELTA_V3_READY");
  pinMode(SOLENOID_PIN, OUTPUT);
  digitalWrite(SOLENOID_PIN, LOW);
  home_reached = false;
  pinMode(ENCODER_A, INPUT_PULLUP);
  pinMode(ENCODER_B, INPUT_PULLUP);
  pinMode(ENCODER_Z, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENCODER_A), encoderA_ISR, RISING);
  attachInterrupt(digitalPinToInterrupt(ENCODER_Z), encoderZ_ISR, RISING);
  baseMaxStepPerSec = (TARGET_RPM * (float)MICROSTEPS) / 60.0f;
  baseAcceleration  = (ACCEL_RPM  * (float)MICROSTEPS) / 60.0f;

  for (int i = 0; i < MOTOR_COUNT; ++i) {
    pinMode(ENA_PINS[i], OUTPUT);
    digitalWrite(ENA_PINS[i], LOW);
    motors[i].setMaxSpeed(baseMaxStepPerSec);
    motors[i].setAcceleration(baseAcceleration);
    motors[i].setCurrentPosition(0);
    motors[i].moveTo(0);
  }
}

void loop() {
  unsigned long now = millis();
  processSerialInput();
  bool busyBefore = motorsBusy();
  for (int i = 0; i < MOTOR_COUNT; ++i) motors[i].run();
  bool busyAfter = motorsBusy();
  if (busyBefore && !busyAfter) {
    settleStartMs = now;
    settling = true;
  }

  if (settling) {
    if (motorsBusy()) {
      settling = false;
      settleStartMs = 0;
    } else if (now - settleStartMs >= SETTLE_MS) {
      if (activeMoveId.length() > 0) {
        String finishedId = activeMoveId;
        Serial.print("DONE ");
        Serial.println(finishedId);

        if (finishedId.endsWith("_S")) {
          home_reached = false;
          setSolenoid(false, "");
        } else if (finishedId.equals("INIT_H") || finishedId.endsWith("_H")) {
          home_reached = true;
          setSolenoid(true, "");
        }

        activeMoveId = "";
      }
      settling = false;
      settleStartMs = 0;
    }
  }

  if (now - lastSpeedUpdateMs >= SPEED_UPDATE_MS) {
    calculateSpeed(SPEED_UPDATE_MS);
    lastSpeedUpdateMs = now;
  }

  if (telemetryEnabled && now - lastTelemetryMs >= TELEMETRY_MS) {
    publishTelemetry();
    lastTelemetryMs = now;
  }
}

void processSerialInput() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;

    if (c == '\n') {
      incomingBuf[incomingPos] = 0;
      if (incomingPos > 0) {
        String cmd = String(incomingBuf);
        cmd.trim();
        handleCommand(cmd);
      }
      incomingPos = 0;
    } else {
      if (incomingPos < (int)sizeof(incomingBuf) - 1) {
        incomingBuf[incomingPos++] = c;
      } else {
        incomingPos = 0;
        while (Serial.available()) {
          char d = (char)Serial.read();
          if (d == '\n') break;
        }
      }
    }
  }
}

void handleCommand(const String &cmd) {
  if (cmd.length() == 0) return;

  String tag = extractTag(cmd);

  if (cmd.equalsIgnoreCase("TELEM_ON") || cmd.startsWith("TELEM_ON")) {
    telemetryEnabled = true;
    printAckWithTag("ACK_TELEM_ON", tag, 2);
    return;
  }
  if (cmd.equalsIgnoreCase("TELEM_OFF") || cmd.startsWith("TELEM_OFF")) {
    telemetryEnabled = false;
    printAckWithTag("ACK_TELEM_OFF", tag, 2);
    return;
  }

  if (cmd.equalsIgnoreCase("STOP") || cmd.startsWith("STOP")) {
    stopAll();
    printAckWithTag("ACK_STOP", tag, 2);
    return;
  }

  if (cmd.equalsIgnoreCase("SOL_ON") || cmd.startsWith("SOL_ON")) {
    if (!home_reached) {
      Serial.println("WARN: SOL_ON ignored (not at HOME)");
      return;
    }
    setSolenoid(true, tag);
    return;
  }

  if (cmd.equalsIgnoreCase("SOL_OFF") || cmd.startsWith("SOL_OFF")) {
    setSolenoid(false, tag);
    return;
  }

  if (cmd.startsWith("SET_POS")) {
    char buf[128];
    cmd.toCharArray(buf, sizeof(buf));
    char *t = strtok(buf, " ");
    t = strtok(NULL, " "); if (!t) return; float a0 = atof(t);
    t = strtok(NULL, " "); if (!t) return; float a1 = atof(t);
    t = strtok(NULL, " "); if (!t) return; float a2 = atof(t);

    motors[0].setCurrentPosition(jointAngleToMotorSteps(0, a0));
    motors[1].setCurrentPosition(jointAngleToMotorSteps(1, a1));
    motors[2].setCurrentPosition(jointAngleToMotorSteps(2, a2));

    motors[0].moveTo(motors[0].currentPosition());
    motors[1].moveTo(motors[1].currentPosition());
    motors[2].moveTo(motors[2].currentPosition());

    printAckWithTag("ACK_SET_POS", tag, 3);
    return;
  }

  if (cmd.startsWith("MOVE_ANGLES")) {
    char buf[128];
    cmd.toCharArray(buf, sizeof(buf));
    char *t = strtok(buf, " ");
    t = strtok(NULL, " "); if (!t) return;
    String id = String(t);

    t = strtok(NULL, " "); if (!t) return; float a0 = atof(t);
    t = strtok(NULL, " "); if (!t) return; float a1 = atof(t);
    t = strtok(NULL, " "); if (!t) return; float a2 = atof(t);
    t = strtok(NULL, " ");
    float speedMult = 1.0f;
    if (t) speedMult = atof(t);

    startMoveAngles(id, a0, a1, a2, speedMult);
    return;
  }

  if (cmd.equalsIgnoreCase("STATUS") || cmd.startsWith("STATUS")) {
    noInterrupts();
    float mm_s = linearSpeed_mm_s;
    long pulses = totalPulses;
    interrupts();

    const float MM_PER_REV   = PI * DRIVE_ROLLER_DIAMETER * 1000.0f;
    const float MM_PER_PULSE = MM_PER_REV / (float)PULSES_PER_REVOLUTION;
    float pos_mm = pulses * MM_PER_PULSE;

    Serial.print("STAT ");
    Serial.print(mm_s, 3);
    Serial.print(" ");
    Serial.print((int)motorsBusy());
    Serial.print(" ");
    Serial.print(pulses);
    Serial.print(" ");
    Serial.print(pos_mm, 3);
    if (tag.length() > 0) { Serial.print(" "); Serial.print(tag); }
    Serial.println();
    return;
  }

  if (ENABLE_SERIAL_PRINTS) {
    Serial.print("CMD_UNKNOWN: ");
    Serial.println(cmd);
  }
}

void startMoveAngles(const String &id, float a0, float a1, float a2, float speedMult) {
  long tsteps[3];
  tsteps[0] = jointAngleToMotorSteps(0, a0);
  tsteps[1] = jointAngleToMotorSteps(1, a1);
  tsteps[2] = jointAngleToMotorSteps(2, a2);

  float setMax = baseMaxStepPerSec * speedMult;

  float accScale = speedMult;
  if (accScale > ACC_MULT_LIMIT) accScale = ACC_MULT_LIMIT;
  float setAcc = baseAcceleration * accScale;

  if (setMax < 1.0f) setMax = baseMaxStepPerSec;

  for (int i = 0; i < MOTOR_COUNT; ++i) digitalWrite(ENA_PINS[i], LOW);

  for (int i = 0; i < MOTOR_COUNT; ++i) {
    motors[i].setMaxSpeed(setMax);
    motors[i].setAcceleration(setAcc);
    motors[i].moveTo(tsteps[i]);
  }

  activeMoveId = id;
  settling = false;
  settleStartMs = 0;

  Serial.print("STARTED ");
  Serial.println(id);
}

void stopAll() {
  for (int i = 0; i < MOTOR_COUNT; ++i) {
    motors[i].stop();
    motors[i].setCurrentPosition(motors[i].currentPosition());
    motors[i].moveTo(motors[i].currentPosition());
  }
  activeMoveId = "";
  settling = false;
  settleStartMs = 0;
}

void publishTelemetry() {
  if (Serial.availableForWrite() < 32) return;
  noInterrupts();
  long revs = totalRevolutions;
  float mm_s = linearSpeed_mm_s;
  float frpm = filteredRPM;
  long pulses = totalPulses;
  interrupts();
  const float MM_PER_REV   = PI * DRIVE_ROLLER_DIAMETER * 1000.0f;
  const float MM_PER_PULSE = MM_PER_REV / (float)PULSES_PER_REVOLUTION;
  float pos_mm = pulses * MM_PER_PULSE;

  Serial.print("ENC ");
  Serial.print(mm_s, 3);
  Serial.print(" ");
  Serial.print(rpmAB, 2);
  Serial.print(" ");
  Serial.print(frpm, 2);
  Serial.print(" ");
  Serial.print(revs);
  Serial.print(" ");
  Serial.print(pulses);
  Serial.print(" ");
  Serial.print(pos_mm, 3);
  Serial.println();
}

void encoderA_ISR() {
  static unsigned long lastMicros = 0;
  unsigned long now = micros();
  if (now - lastMicros < 80UL) return;

  if (digitalRead(ENCODER_B) == LOW) {
    pulseCount++;
    pulsesSinceLastZ++;
    totalPulses++;
  } else {
    pulseCount--;
    pulsesSinceLastZ--;
    totalPulses--;
  }
  lastMicros = now;
}

void encoderZ_ISR() {
  static unsigned long lastZDebounce = 0;
  unsigned long now = micros();
  if (now - lastZDebounce < 500UL) return;

  if (lastZMicros != 0) revolutionTimeMicros = now - lastZMicros;
  else revolutionTimeMicros = 0;
  lastZMicros = now;

  totalRevolutions++;
  zSeen = true;

  pulsesSinceLastZ = 0;
  lastZDebounce = now;
}

void calculateSpeed(unsigned long elapsedMs) {
  long signedPulses = 0;
  unsigned long revTime = 0;
  bool zPulseLocal = false;
  unsigned long lastZCopy = 0;

  noInterrupts();
  signedPulses = pulseCount;
  pulseCount = 0;
  revTime = revolutionTimeMicros;
  zPulseLocal = zSeen;
  zSeen = false;
  lastZCopy = lastZMicros;
  interrupts();

  long absPulses = labs(signedPulses);
  if (elapsedMs > 0) {
    rpmAB = ((float)absPulses / (float)PULSES_PER_REVOLUTION) * (60000.0f / (float)elapsedMs);
    if (filteredRPM == 0.0f) filteredRPM = rpmAB;
    filteredRPM += ALPHA * (rpmAB - filteredRPM);
  }

  if (zPulseLocal && revTime > 0) {
    rpmZ = 60000000.0f / (float)revTime;
    if (filteredRPM < 60.0f) filteredRPM = rpmZ;
  }

  if (lastZCopy == 0 || (micros() - lastZCopy) > 2000000UL) rpmZ = 0.0f;

  linearSpeed_m_s  = (filteredRPM / 60.0f) * (PI * DRIVE_ROLLER_DIAMETER);
  linearSpeed_mm_s = linearSpeed_m_s * 1000.0f;
}

bool motorsBusy() {
  for (int i = 0; i < MOTOR_COUNT; ++i) {
    if (motors[i].distanceToGo() != 0) return true;
  }
  return false;
}
void setSolenoid(bool on, const String &tag) {
  digitalWrite(SOLENOID_PIN, on ? HIGH : LOW);
  printAckWithTag(on ? "ACK_SOL_ON" : "ACK_SOL_OFF", tag, 2);
}
