% MSPTEMPOTRON(ts, t_i, w, V_thresh, V_rest, tau_m, tau_s) - multi-spike tempotron neuron model
%   ts: time vector
%   t_i: input pattern as cell array of spike times for each synapse
%   w: synaptic efficiencies / weights
%   V_thresh: spike threshold potential
%   V_rest: resting potential
%   tau_m: membrane time constant
%   tau_s: synapse time constant

function [v_t,t_out,t_out_idx,v_unreset,V_thresh, V_rest, V_0, tau_m, tau_s] = MSPTempotron(ts, t_i, w, V_thresh, V_rest, tau_m, tau_s)
   
   % MATLAB vararg parsing boilerplate
   if nargin < 6
      tau_m = 0.020;
   end
   
   if nargin < 7
      tau_s = 0.005; 
   end
   
   
   eta = tau_m/tau_s;
   V_0 = eta^(eta/(eta-1)) / (eta - 1);     % normalizing constant for syn. currents
    
   v_t = zeros(1, length(ts)) + V_rest;     % init V(t) with resting membrane potential 
   t_out= [];                               % output spike times
   t_out_idx = [];                          % indices of ts vector where output spikes occour
   t_sp_idx = 1;
   
   v_t(t_sp_idx:end) = (0 .* v_t(t_sp_idx:end));    % membrane potential V(t)
       
   % simulate neuron
   for i=1:length(w)
       v_sub = msp_tempotron_kernel(ts, t_i{i}, tau_m, tau_s, V_0);
       v_t = v_t + (w(i).*v_sub);    
   end
   
   v_unreset = v_t; % save unresetted membrane potential
   
   % determine output spike times & perform soft-reset of V(t)
   while (~isempty(t_sp_idx))
       % reached V_threshold, soft-reset & emit spike time
        above = v_t > V_thresh; % 1 and 0 for event / non event
        crossings = diff(above);
        idx = find(crossings>0)+0;
        t_sp_idx = idx(idx ~= t_sp_idx);
    
        if (~isempty(t_sp_idx))
            t_sp_idx = t_sp_idx(1);
            t_out = unique([t_out ts(t_sp_idx)]);
            t_out_idx = unique([t_out_idx t_sp_idx]);
            v_reset = (V_thresh .* exp(-(ts(t_sp_idx:end)- ts(t_sp_idx))/tau_m));% + (v_t(t_sp_idx) - V_thresh);
            v_t(t_sp_idx:end) = v_t(t_sp_idx:end) - v_reset;
        end
   end
end


% kernel to compute synaptic input current
function [v] = msp_tempotron_kernel(t, t_i, tau_m, tau_s, V_0)
    exp_kernel = @(x, tau) exp(-(x)/tau);
    v = zeros(1, length(t));
    for i=1:length(t_i)
       tmp = heaviside(t-t_i(i)) .* (t-t_i(i));
       v = v + (exp_kernel(tmp, tau_m) - exp_kernel(tmp, tau_s));
    end

    v = v .* V_0;
end

