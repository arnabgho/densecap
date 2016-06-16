require 'hdf5'
require 'nn'
require 'os'

local relation_extractor_loader={}
relation_extractor_loader.__index=relation_extractor_loader

function relation_extractor_loader.create(input_h5_file , batch_size , split_fractions , seq_length , feat_dim)
    local self={}
    setmetatable(self,relation_extractor_loader)
    self.h5file=hdf5.open(input_h5_file,'r')
    self.sz=self.h5file:read('data'):dataspaceSize()
    self.total=sz[1]
    self.split_quantities={ math.floor(self.total*split_fractions[1]) ,   math.floor(self.total*split_fractions[2]) ,  math.floor(self.total*split_fractions[3])   }
    self.num_batches = { math.floor(self.split_quantities[1]/self.batch_size) , math.floor(self.split_quantities[1]/self.batch_size) , math.floor(self.split_quantities[1]/self.batch_size )} 
    self.batch_indices={0,0,0}
    self.start_indices={ 1 , 1+self.num_batches[1] , 1+self.num_batches[1]+self.num_batches[2]  }
    self.seq_length=seq_length
    self.feat_dim=feat_dim
    self.batch_size=batch_size
end

function relation_extractor_loader:next_batch(split_index) 
    if self.num_batches(split_index)==0 then
        print("Wrong Split Index , this split doesn't have any data")
        os.exit()
    else
        if self.batch_indices[split_index]==self.num_batches[ split_index ] then self.batch_indices[split_index]=0 end
        return self.h5file:read('data'):partial(  { self.start_indices[split_index] + self.batch_indices[split_index]*self.batch_size ,   (self.batch_indices[split_index]+1)*self.batch_size } , {1,self.seq_length} , {1,feat_dim}  )  , 
               self.h5file:read('labels'):partial(   { self.start_indices[split_index] + self.batch_indices[split_index]*self.batch_size ,   (self.batch_indices[split_index]+1)*self.batch_size } )     
    end

end
