function results = run_codeVggSiamAgainTemslcTrain3SelAllTrain4STrSM0tar2(seq, res_path, bSaveImage, training_num, epoch_num, gpu_id, num_appearances,tar_threshold)
% clear all;

%Adding necessary paths
addpath(genpath('tracking'));
addpath(genpath('utils'));
addpath(genpath('matconvnet-1.0-beta23'));
vl_setupnn;



base_path = '/data/testing_dataset/OTB2015_Matlab/';
video_path = choose_video(base_path);
if isempty(video_path), return, end
[img_files, pos, target_sz, ground_truth, video_path] = ...
    load_video_info(base_path, video_path);

seq.init_rect = ground_truth(1,:);
seq.s_frames = strcat(video_path, img_files);
seq.name = video_path;
gpu_id = '1';

gpuDevice(str2num(gpu_id));
seq.format = 'otb';
config.seq = seq;
config.display = 0;
config.showtem = 0;
config.num_appearances = str2num(num_appearances);
config.tar_threshold = str2num(tar_threshold);
results = vgg_siam_tracking(config);

rmpath(genpath('tracking'));
rmpath(genpath('utils'));
rmpath(genpath('matconvnet-1.0-beta23'));
%% Calculate performance

% [distance_precision, area, average_center_location_error] = ...
%     compute_performance_measures(result, seq_info.gt);

% fprintf('\n%f fps is %f\n',area, fps);
end


       



