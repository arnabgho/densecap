--[[
TODO : Use the modules to get the activations from the labeled boxes 
TODO : Define the LSTM module that learns the relationship between the subject and the object relationships from the ground truth bounding boxes
TODO : Load the json files corresponding to the relationship file relationships.json and pass the images corresponding to them
TODO : Learn the LSTM module and create the visualizations
--]] 

require 'nn'
require 'rnn'
require 'cjson'
require 'cudnn'

opt = lapp[[
--rnn_size                  (default 512)                              The size of the RNN
--vocab_size                (default 1000)                             The size of the vocabulary    
--num_relations             (default 100)                              The size of the softmax for the number of relations
--feature_size              (default 4096)                             The dimension of the encoded 
--model                     (default "lstm")                           The Recurrent Model to use
--dropout                   (default 0.5)                              The dropout ratio
--batch_size                (default 32)                               Batch Size 
--rmsprop_lr                (default 2e-3)                             RMSPROP Learning Rate 
--rmsprop_dr                (default 0.95)                             RMSPROP Decay Rate
--gpu                       (default 0)                                The GPU ID to use
--log                       (default 'data/extract_features')          The folder in which the trained model is to be saved
--saveFreq                  (default 10)                               The Frequency at which the network is to be saved
--num_epochs                (default 100)                              The Maximum number of epochs of training required
--input_h5                  (default 'data/h5s/encoding_boxes_gt.h5')  The input h5 file
--relationship_map          (default 'data/processed_jsons/relationship_map_gt.json' ) The map containing the predicate-int mapping
--inverse_map               (default 'data/processed_jsons/relationship_inverse_map_gt') The map contining the int-predicate mapping
--seq_length                (default 2)                                 The sequence length of data
--train                     (default 0.9)                               The training fraction
--test                      (default 0.1)                               The testing fraction
]]

print(opt)


-- Create the inverse mapping
local rmap_file=io.open(opt.relationship_map,'r')
local rmap_json_text=rmap_file:read()
local rmap=cjson.decode(rmap_json_text)

local inv_map={}
opt.num_relations=0
for k,v in rmap_json do 
    inv_map[v]=k 
    opt.num_relations=opt.num_relations+1 
end

local inv_map_text=cjson.encode(inv_map)
local inv_map_file=io.open(opt.inverse_map,'w')
inv_map_file:write(inv_map_text)
inv_map_file:close()


---------------------------

if opt.gpu==0 then
    opt.gpu=true
else
    opt.gpu=false
end
-- Load the loader file
local relation_extractor_loader=require 'relation_extractor_loader'

local loader=relation_extractor_loader.create( opt.input_h5 , opt.batch_size , {opt.train,opt.test,1-opt.train-opt.test} , 2 ,feature_size) 
-- LSTM network 

if opt.model=='lstm' then
    rnn=nn.LSTM(opt.feature_size,opt.rnn_size)
elseif opt.model=='gru' then
    rnn=nn.GRU(opt.feature_size,opt.rnn_size)
elseif opt.model=='rnn' then
    rnn=nn.RNN(opt.feature_size,opt.rnn_size)
end

relation_extractor=nn.Sequential()
relation_extractor:add(nn.SplitTable(1,2))
local seq1=nn.Sequencer(rnn)

seq1:remember()

relation_extractor:add(seq1)

local seq2=nn.Sequencer( nn.Dropout( opt.dropout  )  )

seq2:remember()

relation_extractor:add(seq2)

relation_extractor:add(nn.SelectTable(-1))

relation_extractor:add( nn.Linear( opt.rnn_size , opt.num_relations  )   )

relation_extractor:add( nn.LogSoftMax())

criterion=nn.ClassNLLCriterion()

parameters_re,gradParameters_re=relation_extractor:getParameters()

rmspropState_re={learningRate=opt.rmsprop_lr, alpha=opt.rmsprop_dr}


local function feval(x)
    if x~= parameters_re then
        parameters_re:copy(x)
    end
    gradParameters_re:zero()

    collectgarbage()

    ---------------------get minibatch ------------------------
    local x ,y = loader:next_batch(1)

    ---------------------forward pass-------------------------
    
    local loss=0
    local pred_y=relation_extractor:forward(x)
    loss=loss+criterion:forward(pred_y,y)

    local dloss_do=criterion:backward(pred_y,y)
    relation_extractor:backward(x,dloss_do)

    return loss,gradParameters_re
end

local function train_relation_extractor() 
    local num_train=loader:totalTraining()
    local batch_size=opt.batchSize
    for epoch = 1,opt.num_epochs do
        local num_iter_per_epoch=math.floor(num_train/batch_size)
        for i in 1,num_iter_per_epoch do
            local _,loss_rnn=optim.rmsprop( feval , parameters_re , rmspropState_re )
        end

        if epoch % opt.saveFreq==0 then
            local file='re_rnn_epoch'..epoch
            local filename=path.concat(opt.save,file)
            os.execute('mkdir -p '.. sys.dirname(filename))
            if paths.filep(filename) then
                os.execute('mv '..filename .. ' ' .. filename .. '.old') 
            end
            print('<trainer> saving network to '..filename)
            torch.save(filename,{model=relation_extractor})
        end
    end
end
    

