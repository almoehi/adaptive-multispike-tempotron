% FIT_MSP_TEMPOTRON(ts, trials, labels, w, V_thresh, V_rest, tau_m, tau_s, lr, n_iter, optimizer, fn_target)
%  train multi-spike tempotron on given trials and labels
%   ts: time vector
%   trials: cell array of trials. Each entry is a cell array of input spike times
%   labels: labels (cumulative number of output spikes) for each trial
%   w: synaptic efficiencies / weights
%   V_thresh: spiking threshold of neuron model (see MSPTempotron)
%   V_rest: resting potential of neuron model (see MSPTempotron)
%   tau_m: membrane time constant of neuron model (see MSPTempotron)
%   tau_s: synapse time constant of neuron model (see MSPTempotron)
%   lr: learning rate parameter
%   n_iter: total number of iterations performed
%   optimizer: one of 'sgd', 'adagrad', 'rmsprop', 'adam'
%   fn_target: function handle to custom error function with signature fn(sample_idx, t_out, target_cum_reward)

function [w, t_crit, dv_dw, errs, outputs, w_hist, anneal_lr, t_iter] = fit_msp_tempotron(ts, trials, labels, w, V_thresh, V_rest, tau_m, tau_s, lr, n_iter, optimizer, fn_target)
    
    if nargin < 12
        fn_target = [];
    end
    
    if nargin < 11
       optimizer = 'rmsprop'; 
    end
    
    dataFormatType = iscell(trials{1});
    if dataFormatType == 0
        % this means, data is formated as cell array with spikes times as
        % columns (per synapse)
        N_syn = size(trials(1,:), 2);
    else
        N_syn = length(trials{1});
    end
    
    errs = [];
    outputs = zeros(1, size(trials, 1));
    d_momentum = zeros(1, N_syn);
    t_crit = 0;
    dv_dw = [];
    w_hist = [];
    grad_cache = zeros(1, N_syn); %adagrad / RMSprop gradient cache
    eps = 10^-6;
    momentum_mu = 0.99;    % momentum hyper param
    rms_decay_rate = 0.99; % RMSprop leak
    anneal_lr = lr;
    t_iter = n_iter;
    
    shuffle_idx = randperm(size(trials, 1));
    profile_start = tic;
    for i=1:size(trials,1)
        % determine format of pattern
        if dataFormatType == 0
            pattern = cell(trials(i,:));
        else
            pattern = trials{i};
        end
        
        target = labels(i);
        
        if mod(i, 10) == 0
           tElapsed = toc(profile_start);
           disp(sprintf('   trial %d [%.3f sec]', i, tElapsed)); 
           profile_start = tic;
        end
        
        [v_t, t_out, t_out_idx, v_unreset, ~, ~, V_0, tau_m, tau_s] = MSPTempotron(ts, pattern, w, V_thresh, V_rest, tau_m, tau_s);
        outputs(i) = length(t_out);
        % keep track on errors
        if (~isempty(fn_target))
            %err = fn_target(shuffle_idx(i), t_out, labels(shuffle_idx(i))) - outputs(shuffle_idx(i));
            err = fn_target(i, t_out, target);
            %disp(sprintf('   err=%d out=%d target=%d', err, outputs(shuffle_idx(i)), labels(shuffle_idx(i))));
        else
            err = target - length(t_out);
        end
        
        if (any(isnan(v_t)))
           error('NaNs !!!'); 
        end
        
        if (err ~= 0) % perform weight updates only on error trials
            errs = [errs err];
            
            [pks, pks_idx, t_crit, d_w, dw_dir, dv_dw] = msp_grad(V_0, V_thresh, pattern, w, ts, v_t, v_unreset, t_out, t_out_idx, err, tau_m, tau_s);
                        
            if strcmpi(optimizer, 'adagrad') == 1
                % ADAgrad optimizer
                %disp('** adagrad');
                grad_cache = grad_cache + d_w.^2;
                delta = (((dw_dir * lr) .* d_w) ./ (sqrt(grad_cache) + eps));
            elseif strcmpi(optimizer, 'rmsprop') == 1
                % RMSprop
                %disp('** RMSprop');
                grad_cache = rms_decay_rate .* grad_cache + (1 - rms_decay_rate) .* d_w.^2;
                delta = (((dw_dir * lr) .* d_w) ./ (sqrt(grad_cache) + eps));
            elseif strcmpi(optimizer, 'momentum') == 1
                %disp('** Momentum');
                % Momentum
                d_momentum = ((dw_dir * lr) .* d_w) + (momentum_mu .* d_momentum);
                delta = d_momentum; 
            
            else
                %default: vanilla SGD
                %disp('** SGD');
                delta = ((dw_dir * lr) .* d_w); % regular gradient-based learning
            end
            
            % update weights
            w = w + delta;
            w_hist = [w_hist; w];
        end
    end
end