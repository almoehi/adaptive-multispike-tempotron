clear;
close all;

% sample script that reproduces convergence figures from our paper
% for both optimizers (adaptive & Momentum)

modelName = 'gamma_5-inhomogbgnoise-trials-250-syn-500-feat-9';

trainSetFilename = ['data/', modelName, '-train.mat'];
validationSetFilename = ['data/', modelName, '-validation.mat'];

N_sim = 2; % no of indep. simulations to run
seeds = randperm(500, N_sim);

% load validation set
load(validationSetFilename);
data_validation = data;

% layout of data cell - each row is a sample
%data{n,1} = trial_times;      % bg noise + patterns - that is the actual trial 
%data{n,2} = bg_times;         % bg noise only
%data{n,3} = clue_times;       % all patterns only
%data{n,4} = clue_onset;       % time of each pattern onset
%data{n,5} = merged_rewards;   % rewards for each pattern (sum == training target)
%data{n,6} = T;                % total trial duration in sec.
%data{n,7} = T_features;       % duration of a single feature in sec.
%data{n,8} = dt;               % time resolution

% to use for your own data, data_train and data_validation can also be 
% just cell arrays of shape: (n_samples, n_synapses)
% where the 2nd dimension is a cell array of size n_synapses
% where each entry contains a vector of spike-times for that synapse
% e.g. for 3 synapses: sp = cell({[0.023],[],[0.56 1.234 4.246]});
% this data format can easily be constructed from 
% python and saved as MAT file using scipy.io

% load training set - same layout
load(trainSetFilename);
data_train = data;

% fitting parameters
N_trials = length(data);
N_syn = length(data{1,1});
dt = 1/1000;        % integration time-step for discrete-time simulation. This can be improved.
T = data{1,6}(1);   % duration of a single trial (can also vary from trial to trial)
lr = 0.001;         % learning rate
ts = 0:dt:T;        % time axis for discrete-simulation for trial duration T

% neuron model
tau_m = 0.015;
tau_s = 0.005;
V_thresh = 1;
V_rest = 0;

n_epochs = 10;  % training epochs
train_trials = data_train(:,1);
train_targets = [];

validation_trials = data_validation(:,1);
validation_targets = [];

% compute training targets as sum of individual rewards
for i=1:N_trials
   train_targets(i) = sum(data_train{i,5}); 
end

for i=1:length(validation_trials)
   validation_targets(i) = sum(data_validation{i,5}); 
end

lr_momentum = 0.001;
lr_rmsprop = 0.001;

all_losses_momentum = [];
all_losses_rmsprop = [];
validations_momentum = [];
validations_rmsprop = [];
weights_momentum = [];
weights_rmsprop = [];

optimizer_1 = 'rmsprop';
optimizer_2 = 'momentum';
optimizer_1_name = 'RMSProp';
optimizer_2_name = 'Momentum';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% run learning vor several epochs and several random seeds
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k=1:length(seeds)
    n_iter = 0;
    n_iter_momentum = n_iter;
    n_iter_rmsprop = n_iter;
    rng(seeds(k));
    w_momentum = normrnd(0, 1 / N_syn, 1, N_syn);
    %w_momentum = normrnd(0, .01, 1, N_syn);
    w_rmsprop = w_momentum;
    losses_momentum = zeros(1, n_epochs);
    losses_rmsprop = zeros(1, n_epochs);
    validation_momentum = zeros(1, n_epochs);
    validation_rmsprop = zeros(1, n_epochs);
    
    for i=1:n_epochs
        [w_momentum, ~, ~, errs_momentum, pred_momentum, ~, ~, n_iter_momentum] = fit_msp_tempotron(ts, train_trials, train_targets, w_momentum, V_thresh, V_rest, tau_m, tau_s, lr_momentum, n_iter_momentum, optimizer_1);
        loss_momentum = mean(abs(pred_momentum-train_targets));
        losses_momentum(1,i) = loss_momentum;
        [mean_val_loss, ~] = validate_msp_tempotron(ts, validation_trials, validation_targets, w_momentum, V_thresh, V_rest, tau_m, tau_s);
        validation_momentum(1,i) = mean_val_loss;
        
        if (isempty(errs_momentum)) % all zeros => no errors, converged
            disp(sprintf('MOMENTUM learning converged after %d epochs', i));
            break;
        end

        if (mod(i, 1) == 0)
            disp(sprintf('sim=%d | epoch=%d | lr=%.4f | training loss: %.3f | validation loss: %.3f', k, i, lr_momentum, loss_momentum, mean_val_loss));
        end
    end
    
    weights_momentum = [weights_momentum; w_momentum];
    all_losses_momentum = [all_losses_momentum; losses_momentum];
    validations_momentum = [validations_momentum; validation_momentum];
    
    for i=1:n_epochs
        [w_rmsprop, ~, ~, errs_rmsprop, pred_rmsprop, ~, ~, n_iter_rmsprop] = fit_msp_tempotron(ts, train_trials, train_targets, w_rmsprop, V_thresh, V_rest, tau_m, tau_s, lr_rmsprop, n_iter_rmsprop, optimizer_2);
        loss_rmsprop = mean(abs(pred_rmsprop-train_targets));
        losses_rmsprop(1,i) = loss_rmsprop;
        
        [mean_val_loss, ~] = validate_msp_tempotron(ts, validation_trials, validation_targets, w_rmsprop, V_thresh, V_rest, tau_m, tau_s);
        validation_rmsprop(1,i) = mean_val_loss;
        
        if (isempty(errs_rmsprop)) % all zeros => no errors, converged
            disp(sprintf('RMSProp learning converged after %d epochs', i));
            break;
        end

        if (mod(i, 1) == 0)
            disp(sprintf('sim=%d | epoch=%d | lr=%.4f | training loss: %.3f | validation loss: %.3f', k, i, lr_rmsprop, loss_rmsprop, mean_val_loss));
        end
    end
    
    weights_rmsprop = [weights_rmsprop; w_rmsprop];
    all_losses_rmsprop = [all_losses_rmsprop; losses_rmsprop];
    validations_rmsprop = [validations_rmsprop; validation_rmsprop];
    
end

disp('saving model(s)');
save(sprintf('msp-model_%s.mat', modelName), 'data_train', 'data_validation', 'seeds', 'all_losses_rmsprop', 'all_losses_momentum', 'validations_rmsprop', 'validations_momentum', 'weights_momentum', 'weights_rmsprop');

% plot convergence curve(s)
fig = figure();

ax = subplot(2, 2, 1);
links = [ax];
hold on;
for i=1:length(seeds)
   plot(ax, 1:n_epochs, all_losses_momentum(i,:), 'k');
end
ylabel(ax, 'train error');
title(ax, sprintf('(k=%d, N=%d)', length(seeds), N_syn));
xlabel(ax, 'training epoch');
legend({optimizer_1_name});

ax = subplot(2, 2, 2);
links = [links; ax];
hold on;
for i=1:length(seeds)
   plot(ax, 1:n_epochs, all_losses_rmsprop(i,:), 'k');
end
ylabel(ax, 'train error');
title(ax, sprintf('(k=%d, N=%d)', length(seeds), N_syn));
xlabel(ax, 'training epoch');
legend({optimizer_2_name});

linkaxes(links, 'y');

ax = subplot(2, 2, 3);
links = [ax];
hold on;
for i=1:length(seeds)
    plot(ax, 1:n_epochs, validations_momentum(i, :), 'k');
end
ylabel(ax, 'generalization error');
title(ax, sprintf('(k=%d, N=%d)', length(seeds), N_syn));
xlabel(ax, 'training epoch');
legend({optimizer_1_name});

ax = subplot(2, 2, 4);
links = [links; ax];
hold on;
for i=1:length(seeds)
    plot(ax, 1:n_epochs, validations_rmsprop(i, :), 'k');
end
ylabel(ax, 'generalization error');
title(ax, sprintf('(k=%d, N=%d)', length(seeds), N_syn));
xlabel(ax, 'training epoch');
linkaxes(links, 'y');
legend({optimizer_2_name});