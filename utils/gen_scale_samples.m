function [feats]=gen_scale_samples(net_feat, img, target_loc, scales, support_sz)
% this function generates the different scale samples based on the input
% parameters
% Input:
%   net_feat    -       feature extraction net
%   img         -       original image
%   target_loc  -       position of the target  [top_left_x top_left_y width height]
%   scales      -       scale ratio
%   support_sz  -       input size

% Output:
%   feats       -       features of scaled samples
img= single(img)-128;
center = target_loc(1:2) + floor(target_loc(3:4)/2);
num_scales = size(scales,2);
szs = bsxfun(@times, target_loc(3:4), scales');
sizes =mat2cell([repmat(center, num_scales,1), szs], ones(1,num_scales),4);

patches = cellfun(@(loc) get_subwindow(img, loc([2 1]), loc([4 3])),...
                                sizes, 'uniformoutput', false);
re_patches = cellfun(@(patch) imresize(patch, [support_sz(1) support_sz(2)],'bilinear','antialiasing',false),...
                                patches,'uniformoutput', false);
patches_mat = cell2mat( reshape(re_patches,1,1,1,num_scales) );

net_feat.eval({'input',gpuArray(patches_mat)});

feats = net_feat.vars(end).value;

end