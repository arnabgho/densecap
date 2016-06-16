import argparse, os, json, string
from collections import Counter
from Queue import Queue
from threading import Thread, Lock

from math import floor
import h5py
import numpy as np
from scipy.misc import imread, imresize

import os
"""
Process the image folder and create an h5 file using the Hadoop File System

Steps to be implemented :
    Read the relationships json file and select 10000 images which are present in the dataset using a try catch statement
    Store the 10000 images in an h5 file and the 10000 relationships in separate json file so that
"""
xwasbad = 0
ywasbad = 0
wwasbad = 0
hwasbad = 0


def read_corresponding_images( relationship_json , image_root , max_limit ):
    valid_images=[]
    valid_relationships=[]
    for i in range(len(relationship_json)):
        if len(valid_images)==max_limit :
            break
        image_id=str(relationship_json[i]['id'])
        #print("Image File:")
        #print( image_root +"/" +image_id + '.jpg' )
        if os.path.isfile( image_root+"/" + image_id + '.jpg'   ):
            if os.path.getsize(  image_root +"/"+ image_id + '.jpg'   )>0:
                valid_images.append( image_id  )
                valid_relationships.append( relationship_json[ i ]  )

    print("Number of valid images: ")
    print(len(valid_images))
    return valid_images,valid_relationships



def clamp_toimage(x,y,w,h,scale,image_size):
    global xwasbad
    global ywasbad
    global wwasbad
    global hwasbad
    x= round( scale*(x - 1)+1)
    y= round( scale*(y - 1)+1)
    w= round( scale*(w ))
    h= round( scale*(h ))

    # clamp to image
    if x<1: x=1
    if y<1: y=1
    if x>image_size-1:
        x=image_size-1
        xwasbad +=1
    if y>image_size-1:
        y=image_size-1
        ywasbad+=1
    if x+w>image_size :
        w=image_size-x
        wwasbad+=1
    if y+h>image_size:
        h=image_size-y
        hwasbad+=1

    return x,y,w,h



def recompute_box_dims( image_size , valid_relationships , image_root ):
    """
        Recompute the bounding boxes for each of the relationship boxes
        and return the cleaned up bounding boxes
    """
    for i in range( len( valid_relationships ) ):
        image_id=str(valid_relationships[i]['id'])
        image_data=imread(image_root+'/'+image_id+'.jpg')
          # handle grayscale
        if image_data.ndim == 2:
            image_data = image_data[:, :, None][:, :, [0, 0, 0]]
        H,W,D=image_data.shape
        scale=float(image_size) / max(H,W)
        for j in range( len( valid_relationships[i]['relationships']  ) ):

            x,y= valid_relationships[i]['relationships'][j]['object'][ 'x'],  valid_relationships[i]['relationships'][j]['object']['x']
            w,h= valid_relationships[i]['relationships'][j]['object']['w'],  valid_relationships[i]['relationships'][j]['object']['h']

            x_new,y_new,w_new,h_new=clamp_toimage(x,y,w,h,scale,image_size)

            valid_relationships[i]['relationships'][j]['object']['x'],  valid_relationships[i]['relationships'][j]['object']['x']=x_new,y_new
            valid_relationships[i]['relationships'][j]['object']['w'],  valid_relationships[i]['relationships'][j]['object']['h']=w_new,h_new

            x,y= valid_relationships[i]['relationships'][j]['subject']['x'],  valid_relationships[i]['relationships'][j]['subject']['x']
            w,h= valid_relationships[i]['relationships'][j]['subject']['w'],  valid_relationships[i]['relationships'][j]['subject']['h']

            x_new,y_new,w_new,h_new=clamp_toimage(x,y,w,h,scale,image_size)

            valid_relationships[i]['relationships'][j]['subject']['x'],  valid_relationships[i]['relationships'][j]['subject']['x']=x_new,y_new
            valid_relationships[i]['relationships'][j]['subject']['w'],  valid_relationships[i]['relationships'][j]['subject']['h']=w_new,h_new




    return valid_relationships


def create_image_h5( image_size ,  image_root ,h5_file ,  valid_images , num_workers  ):
    """
        Create the h5 file containing all the valid reshaped images
    """
    num_images=len(valid_images)
    shape= ( num_images , 3 , image_size ,image_size)
    image_dset = h5_file.create_dataset('images_relationship',shape,dtype=np.uint8)

    lock=Lock()
    q=Queue()

    for i in range(len( valid_images )):
        filename=os.path.join(image_root,"%s.jpg" % valid_images[i])
        q.put((i,filename))

    def worker():
        while True:
            i,filename=q.get()
            img=imread(filename)
            # handle grayscale
            if img.ndim==2:
                img=img[:,:,None][:,:,[0,0,0]]
            H0,W0 = img.shape[0],img.shape[1]
            img=imresize( img, float(image_size) / max(H0,W0))
            H,W=img.shape[0],img.shape[1]
            #swap rgb to bgr
            r=img[:,:,0].copy()
            img[:,:,0]=img[:,:,2]
            img[:,:,2]=r

            lock.acquire()
            if i%1000==0:
                print 'Writing image %d / %d' % ( i, len(valid_images)   )
            image_dset[i,:,:H,:W]=img.transpose(2,0,1)
            lock.release()
            q.task_done()

    print('adding images to hdf5 ...(this might take a while) ')
    for i in xrange(num_workers):
        t=Thread(target=worker)
        t.daemon=True
        t.start()
    q.join()

def main(opt):
    '''
        use the above defined functions to achieve the end result
    '''
    with open(opt.relationship,'r') as f:
        relationship_json=json.load(f)

    valid_images,valid_relationships=read_corresponding_images(   relationship_json , opt.image_root , opt.max_limit  )

    valid_relationships=recompute_box_dims( opt.image_size , valid_relationships , opt.image_root )

    f=h5py.File(opt.h5_output , 'w')
    create_image_h5( opt.image_size ,  opt.image_root ,f ,  valid_images , opt.num_workers  )
    f.close()

    with open(opt.json_output , 'w') as f:
        json.dump( valid_relationships ,f )


if __name__=='__main__':
    '''
        Settings for the input files and the output files
    '''
    parser=argparse.ArgumentParser()

    parser.add_argument('--relationship',
                default='data/jsons/relationships.json',
                help='The Relationship JSON File')

    parser.add_argument('--image_root',
                default='data/images/VG_100K',
                help='The image root in which the images are present')
    parser.add_argument('--max_limit',
                default=10000, type=int ,
                help='The maximum number of images that are to be considered')
    parser.add_argument('--num_workers',
                default=12,type=int ,
                help='The number of threads that are needed for parallel processing')
    parser.add_argument('--image_size',
                default=720 , type=int,
                help='The image size to which all the images have to be reshaped')

    parser.add_argument('--h5_output',
                    default='data/h5s/images_relationship.h5',
                    help='The output file for the valid images to be written')

    parser.add_argument('--json_output',
                    default='data/processed_jsons/valid_relationships.json',
                    help='The json file in which the valid relationships are written')

    opt=parser.parse_args()
    main(opt)


