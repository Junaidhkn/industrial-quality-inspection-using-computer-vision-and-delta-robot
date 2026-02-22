function [pos, valid] = deltaFK(theta_deg)
% deltaFK_numeric - forward kinematics by matching IK (mm)
% Input: theta_deg = [th1 th2 th3] (deg)
% Output: pos = [x;y;z] (column), valid logical

    theta_target = theta_deg(:)';

    % cost = squared wrapped-angle error (deg^2)
    function c = costFun(p)
        [t1,t2,t3,ok] = deltaIK(p(1), p(2), p(3));
        if ~ok || any(~isfinite([t1,t2,t3]))
            c = 1e9; return;
        end
        err = wrap180([t1,t2,t3] - theta_target);
        c = sum(err.^2);
        % soft penalty: prefer z negative (below base)
        if p(3) > 0
            c = c + 1e6*(p(3)^2);
        end
    end

    % multiple starts (robust)
    starts = [ 0 0 -390;  50 0 -390; -50 0 -390; 0 50 -400; 0 -50 -400 ];
    opts = optimset('Display','off','TolX',1e-7,'TolFun',1e-9,'MaxIter',4000,'MaxFunEvals',4000);

    bestC = inf; bestP = [NaN NaN NaN];
    for k = 1:size(starts,1)
        [p,c] = fminsearch(@costFun, starts(k,:), opts);
        if c < bestC
            bestC = c; bestP = p;
        end
    end

    if isfinite(bestC) && bestC < 1e-3
        pos = bestP(:); valid = true;
    else
        pos = [NaN;NaN;NaN]; valid = false;
    end
end

function a = wrap180(a), a = mod(a+180,360) - 180; end
