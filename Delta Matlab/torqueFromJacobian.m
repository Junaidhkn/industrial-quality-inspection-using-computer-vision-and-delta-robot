function tau_Nm = torqueFromJacobian(J, F_N)
% Converts end-effector force (N) to joint torques (N·m)
% J : 3x3 Jacobian (mm/rad)
% F_N : 3x1 force [Fx;Fy;Fz] in Newtons
% tau_Nm : 3x1 joint torque in N·m

    F = F_N(:);           % ensure column
    tau_Nmm = J' * F;     % N*mm  (since J is mm/rad and F is N)
    tau_Nm  = tau_Nmm / 1000;  % -> N·m
end
