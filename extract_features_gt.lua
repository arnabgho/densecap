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
local cjson=require 'cjson'
require 'densecap.DenseCapModel'
local utils = require 'densecap.utils'
local box_utils = require 'densecap.box_utils'
local unpack=table.unpack

local cmd = torch.CmdLine()
unpack=table.unpack
-- Model options
cmd:option('-checkpoint',
  'data/models/densecap/densecap-pretrained-vgg16.t7')
cmd:option('-image_size', 720)
cmd:option('-rpn_nms_thresh', 0.7)
cmd:option('-final_nms_thresh', 0.4)
cmd:option('-num_proposals', 100)
cmd:option('-boxes_per_image', 100)

cmd:option('-input_txt', '')
cmd:option('-max_images', 0)
cmd:option('-output_h5', 'data/h5s/encoding_boxes_gt.h5')

cmd:option('-gpu', 0)
cmd:option('-use_cudnn', 1)

cmd:option('-relationship_h5','data/h5s/images_relationship.h5')
cmd:option('-valid_relationships','data/processed_jsons/valid_relationships.json')
cmd:option('-relationship_map','data/processed_jsons/relationship_map_gt.json')
cmd:option('-thresh',0.95)


local function run_image(model, img,gt_table, opt, dtype)
    -- Load, resize, and preprocess image  
    img = img:float()
    --print(img:size())
    local B=#gt_table
    local gt_boxes=torch.Tensor(1 , B , 4 ):type(dtype)
    local gt_labels=torch.Tensor(1,B,19):type(dtype)
    for i=1,B do
        gt_boxes[1][i]=gt_table[i]:type(dtype)
    end

    model:training()
    model:setGroundTruth(gt_boxes,gt_labels)
    local H, W = img:size(2), img:size(3)
    local img_caffe = img:view(1, 3, H, W)
    img_caffe = img_caffe:index(2, torch.LongTensor{3, 2, 1})
    local vgg_mean = torch.FloatTensor{103.939, 116.779, 123.68}
    vgg_mean = vgg_mean:view(1, 3, 1, 1):expand(1, 3, H, W)
    img_caffe:add(-1, vgg_mean)

    --model.nets.localization_layer:setImageSize(H,W)
    local output=model:updateOutput(img_caffe:type(dtype))
    local final_gt_boxes=output[6]
    
    local final_boxes_xcycwh=output[4]:float()
    local feats=model.nets.recog_base.output:float()

    local final_boxes=box_utils.xcycwh_to_xywh(final_boxes_xcycwh)
    final_boxes=final_boxes:float()
    return {final_gt_boxes,final_boxes,feats}

    --local boxes_xcycwh, feats = model:extractFeatures(img_caffe:type(dtype))
    --local boxes_xywh = box_utils.xcycwh_to_xywh(boxes_xcycwh)
    --return boxes_xywh, feats
end


local function check_intersection( bbgt , bb  )
    local bi={math.max(bb[1],bbgt[1]), math.max(bb[2],bbgt[2]),
    math.min(bb[3],bbgt[3]), math.min(bb[4],bbgt[4])}
    local iw = bi[3]-bi[1]+1
    local ih = bi[4]-bi[2]+1
    local ov=0
    if iw>0 and ih>0 then
        --print("iw and iw non zero")
        -- compute overlap as area of intersection / area of union
        local ua = (bb[3]-bb[1]+1)*(bb[4]-bb[2]+1)+
        (bbgt[3]-bbgt[1]+1)*(bbgt[4]-bbgt[2]+1)-iw*ih
        --print(ua)
        ov = iw*ih/ua
    end
    return ov
end
local function check_with_gt_box( boxes_gt ,boxes_xywh , num_images, opt )
    --local all_matching_boxes={}
    --for i=1,num_images do
        local matching_boxes={}
        for j = 1,#boxes_gt do
            local ovmax=0
            for k = 1,boxes_xywh:size(1) do
                local ov=check_intersection(boxes_gt[j] , boxes_xywh[k])
               -- print("overlapping percentage")
               -- print(ov)
                if ov > opt.thresh and ov>ovmax then
                    matching_boxes[j]=k
                end
            end
        end
    --    table.insert(all_matching_boxes,matching_boxes)
    --end
    -- Returns the pair ( box_gt_id , boxes_xywh_id ) that overlap maximally with the 
    return matching_boxes
end


local function encode_relationships_image(gt_boxes, boxes_xywh , matching_boxes, all_feats , relationship_image , num_images , feat_dim ,opt )
    local all_relationship={}
    local all_relationship_labels={}
    --for i=1,num_images do
    --    local relationship_image=all_relationship_data[i] -- relationship_image is a triplet of (subject_box_gt_id , object_box_gt_id , relation_id )
     --   local matching_boxes=all_matching_boxes[i]
        for j = 1,#relationship_image do
            if matching_boxes[ relationship_image[j][1] ] == nil or matching_boxes[ relationship_image[j][2]  ] == nil then goto continue end
            local feats=torch.Tensor(2,feat_dim)

           -- print("matching_boxes[ relationship_image[j][1] ]")
           -- print( matching_boxes[ relationship_image[j][1] ])

           -- print("matching_boxes[ relationship_image[j][2] ]")
           -- print( matching_boxes[ relationship_image[j][2] ])

           -- print(" relationship_image[j][1] ")
           -- print( relationship_image[j][1] )

           -- print("relationship_image[j][2] ")
           -- print( relationship_image[j][2] )

           -- print("Ground Truth Coordinates")
           -- print(gt_boxes[ relationship_image[j][1]  ])

           -- print("Matched Coordinates")
           -- print(boxes_xywh[ matching_boxes[ relationship_image[j][1]   ]  ])


            feats[1]=all_feats[matching_boxes[ relationship_image[j][1] ]]
            feats[2]=all_feats[matching_boxes[ relationship_image[j][2] ]]
            feats=feats:float()
            table.insert(all_relationship,feats)
            table.insert(all_relationship_labels,relationship_image[j][3])
            ::continue::
        end
    --end
    return {all_relationship,all_relationship_labels}
end    



local function process_encoded_relationships( opt, all_relationship , all_relationship_labels , feat_dim  )
    num_relations=#all_relationship
    data=torch.Tensor( num_relations , 2 , feat_dim  )
    labels=torch.Tensor( num_relations )

    for i=1,num_relations do
        data[i]=all_relationship[i]
        labels[i]=all_relationship_labels[i]
    end
    -- See how to split the data into batches so that the      
    -- write the data to hdf5 files 
    local data_file=hdf5.open(opt.output_h5,'w')
    data_file:write('/data',data)
    data_file:write('/labels',labels)
    data_file:close()
end
-- Function to get the mapping between the ground truth boxes and the id's of the graph format

local function parse_relationship_json( parsed_json ) -- create the parsed json file for the images that you want to process using python (the subset selection to be done by the python code)
    local dict_relationship={}              -- The dictionary of all the relationship triplet 
    local all_relationship_data={}
    local gt_boxes={}
    local relationship_counter=0
    for i =1,#parsed_json do              -- Iterating over all images
        local dict_object_id={}             -- Contains the mapping between the object id in the image and the global box id
        local gt_boxes_image={}             -- Contains the xywh of the ground truth image boxes 
        local relationship_image={}         -- Contains all the relationships of the image
        local object_counter=0
        for j=1,#parsed_json[i]['relationships'] do       -- Iterating over all relations in an image
            local object=parsed_json[i]['relationships'][j]['object']
            local subject=parsed_json[i]['relationships'][j]['subject']
            local predicate=parsed_json[i]['relationships'][j]['predicate']
            if dict_relationship[predicate]==nil then
                relationship_counter=relationship_counter+1
                dict_relationship[predicate]=relationship_counter
            end
            if dict_object_id[object['id']] == nil then
                object_counter=object_counter+1
                dict_object_id[object['id']] = object_counter
                local box=torch.Tensor({ object['x'] , object['y'] , object['w'] , object['h'] })
                gt_boxes_image[object_counter]=box
            end

            if dict_object_id[subject['id']]==nil then
                object_counter=object_counter+1
                dict_object_id[subject['id']] = object_counter
                local box=torch.Tensor({ subject['x'] , subject['y'] , subject['w'] , subject['h'] })
                gt_boxes_image[object_counter]=box 
            end

            table.insert(relationship_image , {  dict_object_id[subject['id']] ,  dict_object_id[object['id']] ,   dict_relationship[predicate]  } )
        end
        table.insert(gt_boxes,gt_boxes_image)
        table.insert(all_relationship_data,relationship_image)
    end

    return {dict_relationship,all_relationship_data,gt_boxes} 
end



local function main()
    local opt = cmd:parse(arg)
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
    local valid_relationship_json=io.open(opt.valid_relationships,'r')
    local valid_relationship_json_text=valid_relationship_json:read()
    local valid_relationship=cjson.decode(valid_relationship_json_text)
    valid_relationship_json:close()

    --print("type of valid_relationship")
    --print(type(valid_relationship))
    -- process the json file to get the relationships and the bounding boxes for each image
    local dict_relationship,all_relationship_data,gt_boxes=unpack( parse_relationship_json( valid_relationship  ) )

   -- will take about 20GB of memory.
    relationship_images_file=hdf5.open( opt.relationship_h5 , 'r'  )
    --local relationship_images_all=relationship_images_file:read('images_relationship'):all()
    --relationship_images_file:close()
    local sizes=relationship_images_file:read('images_relationship'):dataspaceSize()
    print(sizes)

    --print("images_relationship h5 file read")
    local N = sizes[1]
    local M = opt.boxes_per_image
    local D = 4096 -- TODO this is specific to VG
   -- local all_boxes = torch.FloatTensor(N, M, 4):zero()
   -- local all_feats = torch.FloatTensor(N, M, D):zero()
    -- Use the functions defined above
    -- first parse_relationship_json
    local relationship_mapping=cjson.encode(dict_relationship)
    local relationship_mapping_json=io.open(opt.relationship_map,'w')
    relationship_mapping_json:write(relationship_mapping)
    relationship_mapping_json:close()

    collectgarbage()
    --print("parse_relationship_json")

    --local all_matching_boxes={}
    -- Actually run the model
    local all_relationship={}
    local all_relationship_labels={}
    for i=1,N do
        print(string.format('Processing image %d / %d', i, N))
        if #gt_boxes[i]==0 then goto continue end
        local image=relationship_images_file:read('images_relationship'):partial( {i,i} , { 1, 3 }  , { 1 ,opt.image_size  } , { 1 , opt.image_size  })
        image=image[1]
        local final_gt_boxes,final_boxes,feats=unpack( run_image(model, image,gt_boxes[i] , opt, dtype))
        -- all_boxes[i]:copy(boxes[{{1, M}}])
        -- all_feats[i]:copy(feats[{{1, M}}])
        local matching_boxes=check_with_gt_box(gt_boxes[i] , final_gt_boxes , N , opt)
        --table.insert(all_matching_boxes,matching_boxes)  Don't need this perhaps
        --encode_relationships
        local relationships,relationship_labels=unpack(encode_relationships_image(gt_boxes[i] , final_gt_boxes ,  matching_boxes, feats , all_relationship_data[i] , num_images , D ,opt ))
        if #relationships>0 then
            for k,v in pairs(relationships) do table.insert(all_relationship, v) end
            for k,v in pairs(relationship_labels) do table.insert(all_relationship_labels, v) end    
        end 
        print("Number of Relationships till now")
        print(#all_relationship)
        collectgarbage()
        ::continue::
    end

    collectgarbage()
    -- Use the functions defined above
    -- first parse_relationship_json
    -- process the bounding boxes to see which intersect maximally with the ground truth boxes
    --all_matching_boxes=check_with_gt_boxes( gt_boxes , all_boxes , N , opt )

    --collectgarbage()
    --print("check_with_gt_boxes")
    -- process the matching boxes to get the final vector of feats for the subject object relationships for the relevant boxes
    --all_relationship,all_relationship_labels=unpack(encode_relationships( all_matching_boxes, all_feats , all_relationship_data , num_images , feat_dim ,opt ))

    --collectgarbage()
    --print("encode_relationships")
    -- Pack it all into a hdf5 file
    process_encoded_relationships( opt, all_relationship , all_relationship_labels , D  )
    
    collectgarbage()
    --print("process_encoded_relationships")
    -- Write data to the HDF5 file
    --  local h5_file = hdf5.open(opt.output_h5)
    --  h5_file:write('/feats', all_feats)
    --  h5_file:write('/boxes', all_boxes)
    --  h5_file:close()
end

main()
