require 'torch'
require 'nn'
require 'image'
require 'hdf5'

require 'densecap.DenseCapModel'
local utils = require 'densecap.utils'
local box_utils = require 'densecap.box_utils'

json=require 'cjson'
local cmd = torch.CmdLine()

-- Model options
cmd:option('-checkpoint',
  'data/models/densecap/densecap-pretrained-vgg16.t7')
cmd:option('-image_size', 720)
cmd:option('-rpn_nms_thresh', 0.7)
cmd:option('-final_nms_thresh', 0.4)
cmd:option('-num_proposals', 1000)
cmd:option('-boxes_per_image', 20)

cmd:option('-input_txt', '')
cmd:option('-max_images', 0)
cmd:option('-output_h5', 'vqa_box_feat_train.h5')
cmd:option('-mode','train')
cmd:option('-gpu', 0)
cmd:option('-use_cudnn', 1)
cmd:option('img_input_dir','../HieCoAttenVQA/vt-data/')
cmd:option('img_info_json','../Dynamic-memory-networks-in-Theano/data/vqa_fact_data_prepro.json')

local function run_image(model, img_path, opt, dtype)
  -- Load, resize, and preprocess image
  local img = image.load(img_path, 3)
  img = image.scale(img, opt.image_size):float()
  --print(img)
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


local function main()
  local opt = cmd:parse(arg)
  assert(opt.output_h5 ~= '', 'Must provide -output_h5')
  
  local f=io.open(opt.img_info_json)
  local f_text=f:read("*all")
  local json_data=json.decode(f_text)
  f:close()

  -- Read the text file of image paths
  local image_paths = {}
  if opt.mode=="train" then
    image_paths=json_data.unique_img_train
  end

  if opt.mode=="test" then
    image_paths=json_data.unique_img_test
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
    local boxes, feats = run_image(model,opt.img_input_dir .. image_path, opt, dtype)
    all_boxes[i]:copy(boxes[{{1, M}}])
    all_feats[i]:copy(feats[{{1, M}}])
  end

  -- Write data to the HDF5 file
  local h5_file = hdf5.open(opt.train_output_h5)
  h5_file:write('/images_train', all_feats)
  h5_file:write('/boxes', all_boxes)
  h5_file:close()






end

main()
