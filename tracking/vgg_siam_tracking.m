function [results] = vgg_siam_tracking(config)
% this file is the main interface of the proposed tracking algorithm
% Input:
%       img_list    -    image paths 
%       target_loc  -    target location in the first frame
%       display     -    flag indicates whether show the image result
% Output:
%       results     -    tracking results 

    display = config.display;
    showtem = config.showtem;
    num_appearances = config.num_appearances;
    tar_threshold = config.tar_threshold;
    [seq, init_frame] = get_sequence_info(config.seq);
    config = rmfield(config, 'seq');
    if isempty(init_frame)
        seq.rect_position = [];
        [seq, results] = get_sequence_results(seq);
        return;
    end
    
    temp_pos = seq.init_pos-(seq.init_sz-1)/2;
    target_loc = [temp_pos(1,2) temp_pos(1,1) seq.init_sz(1,2) seq.init_sz(1,1)];


    img = init_frame; if size(img,3)==1; img = cat(3, img, img, img); end 
   
    
    opts.output_size = 125;
    opts.padding = 0.1;
    dataMean(1,1,1:3) = single([123,117,104]);
    opts.data_mean(1, 1, 1:3) = dataMean;
    bbox_mode = 'axis_aligned';
    if strcmp(bbox_mode,'axis_aligned')
        get_bbox = @get_axis_aligned_BB;
    elseif strcmp(bbox_mode,'minmax')
        get_bbox = @get_minmax_BB;
    end
    ground_truth = [target_loc];
    ground_truth_4xy = [ground_truth(:,1),ground_truth(:,2),...
            ground_truth(:,1),ground_truth(:,2)+ground_truth(:,4),...
            ground_truth(:,1)+ground_truth(:,3),ground_truth(:,2)+ground_truth(:,4),...
            ground_truth(:,1)+ground_truth(:,3),ground_truth(:,2)];
    bbox_gt = get_bbox(ground_truth_4xy);
    tar_template = gpuArray(imcrop_pad(single(init_frame), bbox_gt, opts.padding, opts.output_size([1,1])));
    
   
    % load template model
    tem_net = load(config.temnet);
    tem_net = tem_net.net;
    tem_net = dagnn.DagNN.loadobj(tem_net);
    tem_net = remove_layers_from_block(tem_net, 'dagnn.Loss');
    tem_net.move('gpu');
   

    re_scale = 1; max_size = 59; min_size = 44;
    ori_target_sz = sqrt(prod(target_loc(3:4)));
   
    if ori_target_sz>max_size,
        re_scale = max_size/ori_target_sz;
    elseif ori_target_sz < min_size
        re_scale = min_size/ori_target_sz;
    end 
    img = imresize(img,re_scale);  
    target_loc = round(target_loc*re_scale);
   
    img_sz       = size(img);   
    search_scale = 3;
    [search_sz, ~]   = cal_window_sz(max_size, img_sz([2 1]), search_scale); %
    input_sz     = [search_sz, search_sz];
    scale_num = 3;
    switch scale_num
        case 3 
            scales        = [45/47 1 45/43];%[1-2/45 1 1+2/45];%
            scale_weights = [0.99 1 1.005];      
        case 5
            scales        = [1-4/45 1-2/45  1  1+2/45  1 +4/45];
            scale_weights = [0.985 0.988 1 1.005 1.006];
       otherwise error('Undefined scale number!\n');    
    end
    feat_layer = {'relu_43'};
   
    %% model initialization
    feat_model_path = '/imagenet-vgg-verydeep-16.mat';
    [net_feat, ~] = initVGG16Net(feat_model_path);
  
    for var_i = 1:numel(feat_layer)
        net_feat.vars(net_feat.getVarIndex(feat_layer{var_i})).precious = true;
    end
   
    [net_match] = init_model();
   
   
  
   
%% First frame processing
   sw_location = floor([ target_loc([1 2])+target_loc([3 4])/2 - search_sz/2, search_sz, search_sz]);
   [feat_groups, swindow] = get_subwindow_feature(net_feat, img, sw_location, input_sz, feat_layer); 
   [patches, t_loc] = gene_feat_patch(target_loc([3 4]), sw_location, feat_groups);
   t_loc(3:4) = t_loc(3:4) - t_loc(1:2)+1;

   sw_feat_sz = size(feat_groups{1});


   if mod(sw_feat_sz(1)*0.05,2)>1
       feat_pad=floor(sw_feat_sz(1)*0.05)+1;
   else
       feat_pad=ceil(sw_feat_sz(1)*0.05)-1;
   end
   assert(mod(feat_pad,2)==0,'pad is not an even number!\n');
   feat_pad=2;
   b_feat_pad = feat_pad/2;
    patch_template = patches{1} * 5;
%     results(1,:) = target_loc;
    feat_maps = zeros(size(feat_groups{1},1),size(feat_groups{1},2),size(patch_template,3),scale_num,'like',patch_template);
%--show the first frame
   if display    
      figure(2); set(gcf,'Position',[200 300 480 320],'MenuBar','none','ToolBar','none');%axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1]);%
      hd = imshow(img,'initialmagnification','fit'); hold on;
      rectangle('Position', target_loc, 'EdgeColor', [0 0 1], 'Linewidth', 1);    
      set(gca,'position',[0 0 1 1]);
      text(10,10,'1','Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30); hold off; drawnow;   
   end 
%% Tracking loop
%    b_f_sz = floor(filter_sz([2 1])/2); %[x y]


    seq.time = 0;
    [seq, im] = get_sequence_frame(seq);
    tracking_result.center_pos = double(seq.init_pos);
    tracking_result.target_size = double(seq.init_sz);% pos, traget_sz 矩阵中坐标格式
    seq = report_tracking_result(seq, tracking_result); %转化矩阵坐标格式 to 直角坐标格式
    
    group_num = 1; 
    if showtem    
        fig_handle = figure('Name', 'ShowGroup');
        hold on;
        subplot(5,4,group_num); imshow(uint8(tar_template));
        hold off;
    end
    group_sel = 1;
    
    i = 0;rep_group_ind = [];
    while true        
        i = i + 1;
%         fprintf('frame = %d\n',i);
        [seq, im] = get_sequence_frame(seq);
        if isempty(im)
            break;
        end
        img = im;
%        img = imread(img_list{frame_i});
        tic
        img = imresize(img,re_scale);
        if size(img,3)==1; img = cat(3, img, img, img); end 

        [feat_groups] = get_subwindow_feature(net_feat, img, sw_location, input_sz, feat_layer);
 
        feat_maps(:,:,:,2)=feat_groups{1};
        feats1 = imresize_gpu(feat_maps(:,:,:,2), sw_feat_sz(1:2)+feat_pad );
        feats3 = imresize_gpu(feat_maps(:,:,:,2), sw_feat_sz(1:2)-feat_pad );
        feat_maps(:,:,:,1) = feats1(b_feat_pad+1:end-b_feat_pad,b_feat_pad+1:end-b_feat_pad,:);
        feat_maps(b_feat_pad+1:end-b_feat_pad,b_feat_pad+1:end-b_feat_pad,:,3) = feats3;

        feat_groups{1} = feat_maps;
       
        [res_maps_ori] = siam_eval( net_match, patch_template(:,:,:,group_sel), feat_groups);

        res_map_up = imresize(res_maps_ori, sw_location([4 3]), 'bicubic');
       
        res_maps = res_map_up+repmat(hann(sw_location(4))*hann(sw_location(3))',1,1,size(res_maps_ori,3));

        [res_map, scale_ind] = cal_scale(res_maps, scale_weights);

        res_ori = res_maps(:,:,scale_ind);
        [max_h, max_w] = find(res_ori == max(res_ori(:))); %update
       
        target_loc_old = target_loc;
       
        t_loc_c = [target_loc(1:2)+target_loc(3:4)/2, target_loc(3:4) ];
       
        t_loc_c(1:2) = t_loc_c(1:2)+ (gather([max_w(1), max_h(1)]) - sw_location(3:4)/2)*scales(scale_ind);
        t_loc_c(3:4) = t_loc_c(3:4) * scales(scale_ind);
       
        target_loc = [t_loc_c(1:2) - t_loc_c(3:4)/2, t_loc_c(3:4)];
        sw_location(3:4) =  round(sw_location(3:4)*scales(scale_ind));
        sw_location(1:2) = [ t_loc_c(1:2) - sw_location(3:4)/2];
       
%        results(frame_i,:) = target_loc/re_scale;
        pos = [t_loc_c(1,2) t_loc_c(1,1)]/re_scale;
        target_sz = [t_loc_c(1,4) t_loc_c(1,3)]/re_scale;
        tracking_result.center_pos = double(pos);
        tracking_result.target_size = double(target_sz);% pos, traget_sz 矩阵中坐标格式
        seq = report_tracking_result(seq, tracking_result); %转化矩阵坐标格式 to 直角坐标格式
%         seq.time = seq.time + toc(); %在while之前设置 seq.time = 0;
        
        
        % grouping templates
        ground_truth = target_loc/re_scale;
        ground_truth_4xy = [ground_truth(:,1),ground_truth(:,2),...
                ground_truth(:,1),ground_truth(:,2)+ground_truth(:,4),...
                ground_truth(:,1)+ground_truth(:,3),ground_truth(:,2)+ground_truth(:,4),...
                ground_truth(:,1)+ground_truth(:,3),ground_truth(:,2)];
        bbox_gt = get_bbox(ground_truth_4xy);
        tracking_patch = gpuArray(imcrop_pad(single(im), bbox_gt, opts.padding, opts.output_size([1,1])));
        tem_net.eval({'exemplar', tracking_patch, 'instance', tar_template});
        
        cmp_score_tar = tem_net.vars(tem_net.getVarIndex('score_tar')).value;
        tar_ind = numel(cmp_score_tar(cmp_score_tar>0));
        cmp_score_group = tem_net.vars(tem_net.getVarIndex('score_group')).value;
        cmp_score = cmp_score_group;
        num_bg = numel(cmp_score(cmp_score>0));

        if tar_ind > 0
            indi_tar = 1
        end

        if num_bg == 0
            indi_new = 1
        end 

        tar_new = indi_tar + indi_new
        
        if tar_new == 2
            if size(tar_template,4) == num_appearances   %10
                group_num = group_num + 1;
                rep_group_ind = mod(group_num,num_appearances-1)+2; %9
                
                if showtem
                    figure(fig_handle);                
                    hold on;
                    subplot(5,4,rep_group_ind); imshow(uint8(tracking_patch));
                    hold off;
                end
                
                tar_template(:,:,:,rep_group_ind) = tracking_patch;
                % extract template patch 
                [feat_groups] = get_subwindow_feature(net_feat, img, sw_location, input_sz, feat_layer);
                [template_patch, ~] = gene_feat_patch(target_loc([3 4]), sw_location, feat_groups);
                template_patch = imresize(template_patch{1}, [size(patch_template,1), size(patch_template,2)]); 
                patch_template(:,:,:,rep_group_ind) = template_patch; 
            else
                group_num = group_num + 1;
                
                if showtem
                    figure(fig_handle);                
                    hold on;
                    subplot(5,4,group_num); imshow(uint8(tracking_patch));
                    hold off;
                end

                tar_template = vl_nnconcat({tar_template, tracking_patch},4); %cat(4, tar_template, tracking_patch);
                % extract template patch 
                [feat_groups] = get_subwindow_feature(net_feat, img, sw_location, input_sz, feat_layer);
                [template_patch, ~] = gene_feat_patch(target_loc([3 4]), sw_location, feat_groups);
                template_patch = imresize(template_patch{1}, [size(patch_template,1), size(patch_template,2)]); 
                patch_template = vl_nnconcat({patch_template, template_patch},4); 
            end
       
        end
        
        group_sel = 1:numel(cmp_score_group);
        if ~isempty(rep_group_ind)
            group_sel = group_sel(group_sel~=rep_group_ind);
        end

        seq.time = seq.time + toc(); 
        
%% ------show results
       if display             
           figure(2);
           imagesc(img); hold on;

           % show score map
           xs = floor(target_loc_old(1)+target_loc_old(3)/2) + (1:sw_location(3)) - floor(sw_location(3)/2);
           ys = floor(target_loc_old(2)+target_loc_old(4)/2) + (1:sw_location(4)) - floor(sw_location(4)/2);
           % show results map
           resp_handle = imagesc(xs, ys, imresize(res_maps_ori(:,:,scale_ind),sw_location([4 3]))); colormap hsv;
           alpha(resp_handle, 0.4);
           
           % show bounding box
           rectangle('Position', target_loc, 'EdgeColor', [ 0 1 0], 'Linewidth', 1);  
           set(gca,'position',[0 0 1 1]);
           text(10,10,num2str(seq.frame),'Color','y', 'HorizontalAlignment', 'left', 'FontWeight','bold', 'FontSize', 30); 
           
           hold off;  drawnow;
       end 
        
   end
%%
% time=1;
[seq, results] = get_sequence_results(seq);
end

