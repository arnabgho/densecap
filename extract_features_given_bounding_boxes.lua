--[[
TODO : Use the modules to get the activations from the labeled boxes 
TODO : Define the LSTM module that learns the relationship between the subject and the object relationships from the ground truth bounding boxes
TODO : Load the json files corresponding to the relationship file relationships.json and pass the images corresponding to them
TODO : Learn the LSTM module and create the visualizations
--]] 

require 'nn'
require 'rnn'
opt = lapp[[
--rnn_size          (default 512)       The size of the RNN
--vocab_size        (default 1000)      The size of the vocabulary    
--num_relations     (default 100)       The size of the softmax for the number of relations
--feature_size      (default 4096)      The dimension of the encoded 
--model             (default "lstm")    The Recurrent Model to use
--dropout           (default 0.5)       The dropout ratio
--batch_size        (default 32)        Batch Size 
--rmsprop_lr        (default 2e-3)      RMSPROP Learning Rate 
--rmsprop_dr        (default 0.95)      RMSPROP Decay Rate
--gpu               (default 0)         The GPU ID to use
]]

print(opt)

if opt.gpu==0 then
    opt.gpu=true
else
    opt.gpu=false
end

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



