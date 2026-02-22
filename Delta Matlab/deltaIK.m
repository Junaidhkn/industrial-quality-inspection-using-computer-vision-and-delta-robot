function [theta1, theta2, theta3, valid] = deltaIK(x, y, z)
% DELTA IK ROBUST - Inverse Kinematics for Delta Robot with Numerical Stability
% Returns angles in degrees (theta=0 is horizontal) and a validity flag.
%
% This version replaces the unstable geometric solution with a robust
% trigonometric method (solving A*sin(th) + B*cos(th) = K), while
% maintaining the user's unique geometric parameter 't'.

%% Robot geometry (mm)
f = 107;  % Base inradius (center to midpoint of side)
e = 36;   % End-effector inradius
rf = 150; % Upper arm length (bicep, Lb)
re = 450; % Lower arm length (parallelogram, Lp)
sqrt3 = sqrt(3);

%% Geometric Offset Calculation
% *** IMPORTANT: Keeping user's specific formula for compatibility with Simulink ***
t = (2*f - e) * sqrt3 / 3;

%% --- Arm 1 (top, 0 degrees) ---
[theta1, v1] = calcAngleYZ_Robust(x, y, z, t, rf, re);

%% --- Arm 2 (rotated +120°) ---
c120 = cosd(120); s120 = sind(120);
x2 = x*c120 + y*s120;
y2 = -x*s120 + y*c120;
[theta2, v2] = calcAngleYZ_Robust(x2, y2, z, t, rf, re);

%% --- Arm 3 (rotated -120° or +240°) ---
c240 = cosd(240); s240 = sind(240);
x3 = x*c240 + y*s240; 
y3 = -x*s240 + y*c240;
[theta3, v3] = calcAngleYZ_Robust(x3, y3, z, t, rf, re);

%% Combined validity
valid = v1 && v2 && v3;
if ~valid
    theta1 = NaN; theta2 = NaN; theta3 = NaN;
end
end

%% --- Robust 3D Trigonometric Helper Function ---
function [theta_deg, valid] = calcAngleYZ_Robust(x0, y0, z0, t, rf, re)
% Solves the single-arm kinematics using the quadratic solution for sine/cosine.
% Convention: theta = 0 is horizontal (Y-axis), positive theta points up (Z-axis).

    % 1. Compensate for the geometric offset 't'
    y_p = y0 + t;
    
    % 2. Define the coefficients for the equation: A*sin(th) + B*cos(th) = K
    % Note: A and B are swapped compared to standard conventions to align
    % theta=0 with the horizontal arm.
    
    A = 2 * rf * z0;                 % Coefficient for sin(theta)
    B = 2 * rf * y_p;                % Coefficient for cos(theta)
    K = x0^2 + y_p^2 + z0^2 + rf^2 - re^2; % Constant term

    % 3. Check Discriminant (Reachability Check)
    % Discriminant = A^2 + B^2 - K^2
    discriminant = A^2 + B^2 - K^2;

    if discriminant < 0
        % Position is unreachable (negative discriminant)
        theta_deg = NaN;
        valid = false;
        return;
    end
    
    sq_discriminant = sqrt(discriminant);

    % 4. Solve for theta using atan2 identity (Robust)
    % The solution is: theta = alpha - beta (or + beta for the other solution)
    
    % alpha: The phase angle (atan2(A, B))
    alpha = atan2(A, B); 
    
    % beta: The adjustment angle (atan2(sq_discriminant, K))
    % We use the negative sign solution (alpha - beta) which typically 
    % corresponds to the "elbow-in" configuration below the base.
    beta = atan2(sq_discriminant, K); 
    
    % Final angle in radians
    theta_rad = -alpha + beta;
    
    % Convert to degrees
    theta_deg = rad2deg(theta_rad);
    valid = true;
end