% MSP_GRAD(V_0, V_thresh, t_i, w_i, ts, v_t, v_unreset, t_out, t_out_idx, N_output, tau_m, tau_s) - compute gradient theta^* for multi-spike tempotron learning rule
%   V_0: normalizing constant of neuron model (see MSPTempotron)
%   V_thresh: spiking threshold of neuron model
%   t_i: current input pattern as cell array of input spike times for each synapse
%   w_i: synaptic efficiencies / weights
%   ts: time vector
%   v_t: membrane potential of neuron for given input pattern t_i
%   v_unreset: unresetted membrane potential
%   t_out: ouput spike times for given input pattern t_i
%   t_out_idx: indices within ts time vector where output spikes occoured
%   N_output: number of desired output spikes (see below)
%   tau_m: membrane time constant of neuron model (see MSPTempotron)
%   tau_s: synapse time constant of neuron model (see MSPTempotron)
%
% N_output - the number of desired ADDITIONAL output spikes
% if N_output > 0 we want more spikes (search for subthreshold peaks)
% if N_output < 0 we want less spikes (determine smallest peaks in v_unreset 
function [pks, pks_idx, t_crit, d_w, dw_dir, dv_dw] = msp_grad(V_0, V_thresh, t_i, w_i, ts, v_t, v_unreset, t_out, t_out_idx, N_output, tau_m, tau_s)

    t_crit = [];
    dt = ts(2) - ts(1);
    
    if N_output > 0
        % determine theta_star which will produce N_output 
        % additional output spikes
        dw_dir = 1; % direction of weight update
        % want more spikes => increase weights, find N largest voltage peaks in subthreshold
       [pks,pks_idx] = findpeaks(v_t); % this will find also find all output spike times
       pks_idx = setdiff(pks_idx, t_out_idx-1); % remove output spike times from set
       
       pks = v_t(pks_idx);
       [S,I] = sort(pks,'descend');
       %the N-th peak is the voltage which will produce N additional spikes 
       idx = min(N_output, length(S));
       if isempty(S)
          v_crit = v_t(1);
          t_crit = ts(1);
       else
           v_crit = S(idx);
           v_crit_idx = pks_idx(I(idx));       
           t_crit = ts(v_crit_idx);
       end
    elseif N_output < 0 && ~isempty(t_out)
        % determine theta_star which will eliminate N_output 
        % output spikes
        dw_dir = -1; % direction of weight update
        % look for the peak above V_threshold which is 
        % closest to V_threshold in unresetted voltage
        [pks,pks_idx] = findpeaks(v_unreset);
        idx_tmp = find(pks > V_thresh); % we're only interested in peaks above threshold
        
        % edge case
        if isempty(idx_tmp)
            pks_idx = t_out_idx;
            pks = v_t(pks_idx);
        else
            pks = pks(idx_tmp);
            pks_idx = pks_idx(idx_tmp);
        end
        
        [S,I] = sort(pks,'ascend');
        safe_pos = min(abs(N_output), length(S));
        v_crit = S(safe_pos);
        v_crit_idx = pks_idx(I(safe_pos));
        t_crit = ts(v_crit_idx);
    end
    
    
    if (~isempty(t_crit))
        % compute dv_dw at t_crit, which needs to be normalized 
        % by gradients of all previous output spikes
        N_syn = length(w_i);        
        % loop over set of time points which conribute to the gradient
        % that is t_crit (t*) and all output spike times < t_crit
        % this is the set t_x of eq 28
        t_x = [t_crit t_out(t_out < t_crit)];
        
        % temporal deriv. of v(t) before each spike time
        v_dot = zeros(1, length(t_x));
        % the weight derivatives at each spike time
        dv_dw = zeros(N_syn, length(t_x));
        % eq 31 for for each output spike
        dv_dt_hist = zeros(1, length(t_x));
        % eq 29 normalizing constant
        c_tx = zeros(1, length(t_x));
        % eq 23,24 normalizing constants due to gradient dependency on
        % previous gradients
        b_k = zeros(N_syn, length(t_x) - 1);
        a_k = zeros(N_syn, length(t_x) - 1);
        % for numerical purpose
        eps = 10^-12;
        
        for k=1:length(t_x)
            t_max = t_x(k); % current time point of set t_x
            t_out_hist = t_out(t_out < t_max); % output spike history up to t_max
            v_tx = v_t(ts == t_max);           % voltage at current timepoint
            
            % eq 32 - here numerical derivative is used instead
            % add eps to prevent division by 0 later on
            v_dot(k) = ((v_tx - v_t(max(1, find(ts == t_x(k)) - 1)))/dt) + eps;
            
            if (k == 1 && dw_dir < 0)
                v_dot(k) = ((V_thresh - v_t(max(1, find(ts == t_x(k)) - 1)))/dt) + eps; 
            end
            %v_dot(k) = ((v_tx - v_t(find(ts == t_x(k)) - 1))*dt); 
            
            % eq 29
            c_tx(k) = 1 + sum(exp(-(t_max - t_out_hist) / tau_m));
            
            % do computations for each synapse
            for j=1:N_syn
               t_in_hist = [t_i{j}];
               t_in_hist = t_in_hist(t_in_hist < t_max);
                       
               % this is eq. to the simple tempotron learning rule
               psp_err = sum(V_0 .* (exp(-(t_max - t_in_hist)/tau_m) - exp(-(t_max - t_in_hist)/tau_s)));
               v_0_tx = -(psp_err .* w_i(j));
               
               % eq 31 - summation over exp() missing in eq 31 !
               
               dv_dt_hist(k) = (v_0_tx / (c_tx(k)^2)) * (sum(exp((-(t_max - t_out_hist))/tau_m)) / tau_m);
               % eq 30 - in principle eq. to simple tempotron but
               % normalized by some factor as we have multiple output
               % spikes now
               dv_dw(j,k) = (1/c_tx(k)) * psp_err;  
               
               % k == 1 is t_crit but a_k and b_k only depend on outp. spikes
                if (k > 1) 
                    sum_to = k-2;
                    v_dot_factor = v_dot(2:(sum_to+1));
                    dv_dt_hist_factor = dv_dt_hist(2:(sum_to+1));
                    
                    % a_ks are independent of w_i
                    % => all rows will be identical so this could be moved
                    % outside the synapse loop
                    % 
                    a_k(j, k-1) = 1 - sum( (a_k(j, 1:sum_to) ./ v_dot_factor) .* dv_dt_hist_factor );
                    b_k(j, k-1) = -(dv_dw(j,k)) - sum( (b_k(j, 1:sum_to) ./ v_dot_factor) .* dv_dt_hist_factor );
                end
            end     
        end
        
        % finally, construct scaling for graient at t_crit 
        % which recursively depends on all gradients of previous output
        % spikes
        v_dot_factor_ab = v_dot(2:end);
        dv_dt_hist_factor_ab = dv_dt_hist(2:end);
        A_star = (1 - sum((a_k ./ v_dot_factor_ab) .* dv_dt_hist_factor_ab, 2));
        B_star = ((-(dv_dw(:,1))) - sum((b_k ./ v_dot_factor_ab) .* dv_dt_hist_factor_ab, 2));
        
        if (~isempty(A_star(A_star == 0)))
           error('A_start is zero - numeric problem !'); 
        end
        
        d_w = -(B_star ./ (A_star))';
        
        if (~isempty(d_w(d_w > 1000)) || any(isnan(d_w)))
           error('diverging gradient ! |d_w|=%.2f', norm(d_w)); 
        end
    end
end