--[[
-- Set the ground truth boxes at first and then have to modify the forward pass of the densecap/LocalizationLayer.lua
-- Use the densecap/modules/BoxSampler.lua to get the indices into the ground truth boxes and the confidence scores
-- Use the DenseCapModel.lua to see what you have to modify for the training pass so that the model doesn't update its parameters
-- Use extractFeatures function from DenseCapModel to get the 4096 codes for the region proposals 
-- Save it in some h5 file
--]]




-- First try the naive model i.e. extract the features using extract_features and then check whether the ground truth boxes intersect say 90% or more use those features


-- Assume we have the features from extract features
--[[
Inputs: The Images Folder and the Relationships JSON File
Outputs: Write in an h5 file the data required by the LSTM layer
--]]
require 'torch'
require 'nn'
require 'image'
require 'hdf5'
require 'cudnn'

require 'densecap.DenseCapModel'
local utils = require 'densecap.utils'
local box_utils = require 'densecap.box_utils'


local cmd = torch.CmdLine()

-- Model options
cmd:option('-checkpoint',
  'data/models/densecap/densecap-pretrained-vgg16.t7')
cmd:option('-image_size', 720)
cmd:option('-rpn_nms_thresh', 0.7)
cmd:option('-final_nms_thresh', 0.4)
cmd:option('-num_proposals', 1000)
cmd:option('-boxes_per_image', 100)

cmd:option('-input_txt', '')
cmd:option('-max_images', 0)
cmd:option('-output_h5', '')

cmd:option('-gpu', 0)
cmd:option('-use_cudnn', 1)

cmd:option('-relationship_path','relationships.json')



local function run_image(model, img_path, opt, dtype)
  -- Load, resize, and preprocess image
  local img = image.load(img_path, 3)
  img = image.scale(img, opt.image_size):float()
  local H, W = img:size(2), img:size(3)
  local img_caffe = img:view(1, 3, H, W)
  img_caffe = img_caffe:index(2, torch.LongTensor{3, 2, 1}):mul(255)
  local vgg_mean = torch.FloatTensor{103.939, 116.779, 123.68}
  vgg_mean = vgg_mean:view(1, 3, 1, 1):expand(1, 3, H, W)
  img_caffe:add(-1, vgg_mean)

  local boxes_xcycwh, feats = model:extractFeatures(img_caffe:type(dtype))
  local boxes_xywh = box_utils.xcycwh_to_xywh(boxes_xcycwh)
  return boxes_xywh, feats
end


local function check_with_gt_boxes( boxes_gt , boxes_xywh , num_images ):
    local all_matching_boxes={}
    for i=1,num_images do
        local matching_boxes={}
        for j in 1,boxes_gt[i] do
            for k in 1,boxes_xcycwh:size(2) do
                if check_intersection(boxes_gt[i][j] , boxes_xcycwh[i][j]) > opt.thresh then
                    matching_boxes[j]=k
                end
            end
        end
        table.insert(all_matching_boxes,matching_boxes)
    end
    -- Returns the pair ( box_gt_id , boxes_xywh_id ) that overlap maximally with the 
    return all_matching_boxes

local function encode_relationships( all_matching_boxes, all_feats , all_relationship_data , num_images , feat_dim ):
    local all_relationship={}
    local all_relationship_labels={}
    for i=1,num_images do
        local relationship_image=all_relationship_data[i] -- relationship_image is a triplet of (subject_box_gt_id , object_box_gt_id , relation_id )
        for j in 1,#relationship_image do
            if all_matching_boxes[ relationship_image[j][1] ] == nil or all_matching_boxes[ relationship_image[j][2]  ] == nil then goto continue end
            local feats=torch.Tensor(2,feat_dim)
            feats[1]=all_feats[i][all_matching_boxes[ relationship_image[j][1] ]]
            feats[2]=all_feats[i][all_matching_boxes[ relationship_image[j][2] ]]
            table.insert(all_relationship,feats)
            table.insert(all_relationship_labels,relationship_image[j][3])
            ::continue::
        end
    end
    return {all_relationship,all_relationship_labels}
    
-- Function to get the mapping between the ground truth boxes and the id's of the graph format

local function parse_relationship_json( parsed_json ) -- create the parsed json file for the images that you want to process using python (the subset selection to be done by the python code)
    local dict_relationship={}
    local all_relationship_data={}
    local gt_boxes={}
    for i in 1,#parsed_json do              -- Iterating over all images
        local dict_object_id={}       
        local gt_boxes_image={}
        local relationship_image={}
        for j in 1,#parsed_json[i]['relationships'] do       -- Iterating over all relations in an image
            local object=parsed_json[i]['relationships'][j]['object']
            local subject=parsed_json[i]['relationships'][j]['subject']
            local predicate=parsed_json[i]['relationships'][j]['predicate']
            if dict_relationship[predicate]==nil then
                local num_rel=#dict_relationship
                dict_relationship[num_rel+1]=num_rel+1
            end
            if dict_object_id[object['id']] == nil then
                local num_id=#dict_object_id
                dict_object_id[object['id']] = num_id+1
                local box=torch.Tensor({ object['x'] , object['y'] , object['w'] , object['h'] })
                gt_boxes_image[num_id+1]=box
            end

            if dict_object_id[subject['id']==nil then
                local num_id=#dict_object_id
                dict_object_id[subject['id']] = num_id+1
                local box=torch.Tensor({ object['x'] , object['y'] , object['w'] , object['h'] })
                gt_boxes_image[num_id+1]=box 
            end

            table.insert(relationship_image , {  dict_object_id[subject['id'] ,  dict_object_id[object['id'] ,   dict_relationship[predicate]  } )
        end
        table.insert(gt_boxes,gt_boxes_image)
        table.insert(all_relationship_data,relationship_image)
    end
    return {dict_relationship,all_relationship_data,gt_boxes} 
end



local function main()
  local opt = cmd:parse(arg)
  assert(opt.input_txt ~= '', 'Must provide -input_txt')
  assert(opt.output_h5 ~= '', 'Must provide -output_h5')
  
  -- Read the text file of image paths
  local image_paths = {}
  for image_path in io.lines(opt.input_txt) do
    table.insert(image_paths, image_path)
    if opt.max_images > 0 and #image_paths == opt.max_images then
      break
    end
  end
  
  -- Load and set up the model
  local dtype, use_cudnn = utils.setup_gpus(opt.gpu, opt.use_cudnn)
  local checkpoint = torch.load(opt.checkpoint)
  local model = checkpoint.model
  model:convert(dtype, use_cudnn)
  model:setTestArgs{
    rpn_nms_thresh = opt.rpn_nms_thresh,
    final_nms_thresh = opt.final_nms_thresh,
    num_proposals = opt.num_proposals,
  }
  model:evaluate()
  
  -- Set up the output tensors
  -- torch-hdf5 can only create datasets from tensors; there is no support for
  -- creating a dataset and then writing it in pieces. Therefore we have to
  -- keep everything in memory and then write it to disk ... gross.
  -- 13k images, 100 boxes per image, and 4096-dimensional features per box
  -- will take about 20GB of memory.
  local N = #image_paths
  local M = opt.boxes_per_image
  local D = 4096 -- TODO this is specific to VG
  local all_boxes = torch.FloatTensor(N, M, 4):zero()
  local all_feats = torch.FloatTensor(N, M, D):zero()
  
  -- Actually run the model
  for i, image_path in ipairs(image_paths) do
    print(string.format('Processing image %d / %d', i, N))
    local boxes, feats = run_image(model, image_path, opt, dtype)
    all_boxes[i]:copy(boxes[{{1, M}}])
    all_feats[i]:copy(feats[{{1, M}}])
  end

  -- Write data to the HDF5 file
  local h5_file = hdf5.open(opt.output_h5)
  h5_file:write('/feats', all_feats)
  h5_file:write('/boxes', all_boxes)
  h5_file:close()
end

main()
