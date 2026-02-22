%% Delta Robot Workspace with  Torque
clc; clear; close all;

% --- Workspace ranges (mm) ---
xRange = -600:30:600;    % X-axis
yRange = -600:30:600;    % Y-axis
zRange = -600:30:-50;    % Z-axis (downward)

% --- End-effector force (N) ---
Fz = 3.5; % downward force in N

% Upper arm length in meters
rf_m = 150/1000; % 150 mm -> meters

% Preallocate
points = [];
torques = [];

% --- Loop through workspace ---
for xi = xRange
    for yi = yRange
        for zi = zRange
            % Inverse Kinematics
            [theta1, theta2, theta3, valid] = deltaIK(xi, yi, zi);
            
            if valid
                theta = [theta1; theta2; theta3];
                
                % Approximate torque for vertical force
                tau = Fz * rf_m * sind(theta); % N·m
                tau_mag = norm(tau); % magnitude
                
                % Store point and torque
                points = [points; xi, yi, zi]; %#ok<AGROW>
                torques = [torques; tau_mag];
            end
        end
    end
end

% --- Plot reachable workspace ---
figure;
scatter3(points(:,1), points(:,2), points(:,3), 20, torques, 'filled');
xlabel('X [mm]');
ylabel('Y [mm]');
zlabel('Z [mm]');
title('Delta Robot Reachable Workspace with Torque Magnitude (Approx.)');
axis equal;
grid on;
colorbar;
colormap(jet);
view(45,30);
