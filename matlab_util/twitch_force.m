function force = twitch_force(t, P, CT)

force = P .* t .* exp(1-t./CT) ./ CT;

