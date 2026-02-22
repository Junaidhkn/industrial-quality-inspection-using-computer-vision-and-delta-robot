function J = deltaJacobian(theta_deg)
% Jacobian (linear velocities) for Delta Robot
% Inputs: theta_deg = [th1 th2 th3] in degrees
% Output: J (3x3) maps end-effector force (N) to joint torques (N·m)

f = 107; e = 36; rf = 150; re = 450;

th = deg2rad(theta_deg);
sqrt3 = sqrt(3);
t = (2*f - e) * sqrt3 / 3;

alpha = [0, 2*pi/3, -2*pi/3];
J = zeros(3,3);

for i = 1:3
    xi = 0; yi = -t;
    if i == 2 || i == 3
        x_temp = xi*cos(alpha(i)) - yi*sin(alpha(i));
        y_temp = xi*sin(alpha(i)) + yi*cos(alpha(i));
        xi = x_temp; yi = y_temp;
    end
    
    % elbow position
    z_elbow = -rf * sin(th(i));
    y_elbow = yi - rf * cos(th(i));
    x_elbow = xi;
    
    % convert to meters
    r = [-x_elbow; -y_elbow; -z_elbow] / 1000; 
    k = [1;0;0]; % rotation axis
    
    % Jacobian without normalization
    J(:,i) = cross(r, k); 
end
end
