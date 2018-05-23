function [mean_loss, validation_errors, predictions] = validate_msp_tempotron(ts, trials, labels, w, V_thresh, V_rest, tau_m, tau_s)
    validation_errors = zeros(1, size(trials, 1));
    predictions = zeros(1, size(trials, 1));
    dataFormatType = iscell(trials{1});
    
    for j=1:size(trials, 1)
       if dataFormatType == 0
            pattern = cell(trials(j,:));
        else
            pattern = trials{j};
       end
        
       [v_t, t_sp, ~, ~, ~] = MSPTempotron(ts, pattern, w, V_thresh, V_rest, tau_m, tau_s); 
       validation_errors(1, j) = abs(length(t_sp) - labels(j));
       predictions(1,j) = length(t_sp);
    end
    
    mean_loss = mean(validation_errors);
end