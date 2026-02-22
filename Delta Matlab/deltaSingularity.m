function isSing = deltaSingularity(J)
% True if Jacobian is ill-conditioned or invalid
    if isempty(J) || any(isnan(J(:))) || any(~isfinite(J(:)))
        isSing = true;
        return;
    end
    c = cond(J);
    isSing = (c > 1e6) || ~isfinite(c);
end
