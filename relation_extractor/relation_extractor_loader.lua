require 'hdf5'
require 'nn'
require 'os'

local relation_extractor_loader={}
relation_extractor_loader.__index=relation_extractor_loader

function relation_extractor_loader.create(input_h5_file , batch_size , split_fractions , seq_length , feat_dim)
    local self={}
    setmetatable(self,relation_extractor_loader)
    self.batch_size=batch_size
    self.h5file=hdf5.open(input_h5_file,'r')
    self.sz=self.h5file:read('data'):dataspaceSize()
    self.total=self.sz[1]
    self.split_quantities={ math.floor(self.total*split_fractions[1]) ,   math.floor(self.total*split_fractions[2]) ,  math.floor(self.total*split_fractions[3])   }
    self.num_batches = { math.floor(self.split_quantities[1]/self.batch_size) , math.floor(self.split_quantities[2]/self.batch_size) , math.floor(self.split_quantities[3]/self.batch_size )} 
    self.batch_indices={0,0,0}
    self.start_indices={ 1 , 1+self.num_batches[1]*self.batch_size , 1+self.num_batches[1]*self.batch_size+self.num_batches[2]*self.batch_size  }
    self.seq_length=seq_length
    self.feat_dim=feat_dim
    self.batch_size=batch_size

    return self
end

function relation_extractor_loader:next_batch(split_index) 
    if split_index>3 or split_index<1 or self.num_batches[split_index]==0 then
        print("Wrong Split Index , this split doesn't have any data")
        os.exit()
    else
      -- print("start")
      -- print( self.start_indices[split_index] + self.batch_indices[split_index]*self.batch_size)
      -- print("end")
      -- print(self.start_indices[split_index]-1+ (self.batch_indices[split_index]+1)*self.batch_size )
        if self.batch_indices[split_index]==self.num_batches[ split_index ] then self.batch_indices[split_index]=0 end
        local x= self.h5file:read('data'):partial(  { self.start_indices[split_index] + self.batch_indices[split_index]*self.batch_size , self.start_indices[split_index]-1 +  (self.batch_indices[split_index]+1)*self.batch_size } , {1,self.seq_length} , {1,self.feat_dim}  )

        local y= self.h5file:read('labels'):partial(   { self.start_indices[split_index] + self.batch_indices[split_index]*self.batch_size ,   self.start_indices[split_index]-1 +  (self.batch_indices[split_index]+1)*self.batch_size } )     

        self.batch_indices[split_index]=self.batch_indices[split_index]+1
        return x,y
    end
end

function relation_extractor_loader:numSamples(split_index)
    if split_index>3 or split_index<1 then
        print("Invalid Split Index")
        os.exit()
    end
    return self.split_quantities[split_index]
end

return relation_extractor_loader
