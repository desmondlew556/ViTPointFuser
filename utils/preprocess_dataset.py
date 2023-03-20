import tarfile
import os
import shutil

# # open file
# file = tarfile.open('gfg.tar.gz')
  
# # extracting file
# file.extractall('./Destination_FolderName')
  
# file.close()
HOME_DEFAULT_PATH = '/home/FYP/desm0019'
DATASET_DEFAULT_PATH = '/DESM0019'

def decodeFile(compressed_file):
    '''
    Decompress and unarchive files and save in the file's current directory
    '''
    file_ext = ['.tar','.tgz']
    # for checking
    destination_directory,ext = os.path.splitext(compressed_file)
    # get destination directory path
    dir_name = os.path.basename(destination_directory)
    dir_path = os.environ.get('HOME',DATASET_DEFAULT_PATH)+DATASET_DEFAULT_PATH + '/' + dir_name
    print('dir_path: '+dir_path)
    try:
        # check file can be decompressed
        if ext in file_ext:
            print('extracting file %s to %s' % (compressed_file,dir_path))
            file = tarfile.open(compressed_file)
            file.extractall(dir_path)
            file.close()
            if os.path.isdir(dir_path): 
                print('Directory path: %s'%(dir_path))
                # remove compressed file
                os.remove(compressed_file)
                pass
    except Exception as err:
        print('Error occurred')
        print(err)
    print("all done")
    return 

def decodeFileToDirectory(compressed_file_path,dest_dir_path):
    '''
    Same as decodeFile, but full paths for file path and destination directory must be given. E.g. to save all extracted files in /home/fileblobs, give this as the full path Decompress and unarchive files and save in the file's current directory
    '''
    file_ext = ['.tar','.tgz']
    # for checking
    dest_dir_name,ext = os.path.splitext(compressed_file_path)

    if os.path.isdir(dest_dir_path): 
        print("path %s is taken"%dest_dir_path)
        return
    try:
        # check file can be decompressed
        if ext in file_ext:
            print('extracting file %s to %s' % (compressed_file_path,dest_dir_path))
            file = tarfile.open(compressed_file_path)
            file.extractall(dest_dir_path)
            file.close()
            if os.path.isdir(dest_dir_path): 
                print('Directory path: %s'%(dest_dir_path))
                # remove compressed file
                # os.remove(compressed_file)
                pass
    except Exception as err:
        print('Error occurred')
        print(err)
    print("all done")
    return 

def iterateAndDecodeFiles():
    '''
    Find compressed/archived dataset files and preprocess for use.
    '''
    dataset_path = os.environ.get('HOME',DATASET_DEFAULT_PATH)+DATASET_DEFAULT_PATH
    files_list = os.listdir(dataset_path) 
    print(dataset_path)
    print(files_list)
    for i in range(0,len(files_list)):
        compressed_file = files_list[i]
        if not compressed_file.startswith('.'):
            compressed_file_path = os.path.join(dataset_path,compressed_file)
            decodeFile(compressed_file_path)

    return

def moveFilesToNewPath(source_dir,dest_dir):
    '''
    Move files in the source directory to the destination directory for nuscenes. Iterates through the sub directories.
    E.g. to move all folders within folder folder1 to folder folder2, call moveFilesToNewPath(<path to folder1>/folder1,<path to folder2>/folder2)
    '''
    # list all documents
    directory_files = os.listdir(source_dir)
    
    def error_func(src,dest,num_files_moved): 
        print("Shutil failed to copy. unable to move file %s to %s"%(src,dest))
        try:
            print("Attempting to copy.")
            shutil.copy2(src,dest)
        except:
            print("Copying failed...")
        return num_files_moved - 1
        
    
    # go through all documents
    print('moving files from %s to %s'%(source_dir,dest_dir))
    # track number of files iterated and moved
    number_of_files = 0
    num_files_moved = 0
    for i in range(len(directory_files)):
        doc_path = os.path.join(source_dir,directory_files[i])
        
        # recursively enters sub folders and move files
        if os.path.isdir(doc_path):
            new_dest_dir = os.path.join(dest_dir,directory_files[i])
            child_number_of_files,child_num_files_moved = moveFilesToNewPath(doc_path,new_dest_dir)
            number_of_files+=child_number_of_files
            num_files_moved+=child_num_files_moved
        else:
            try:
                if not directory_files[i].startswith('.'):
                    number_of_files +=1 
                    num_files_moved +=1
                    shutil.move(doc_path,dest_dir,lambda src,dest: error_func(src,dest,num_files_moved))
                else:
                    print("skipping hidden file %s"%(doc_path))
            except Exception as e:
                num_files_moved -=1
                print("unable to move file %s to %s"%(doc_path,dest_dir))
    
    return number_of_files,num_files_moved

def moveFolderToNewPath():
    '''
    Move nuscenes folder blobs from source directory to the destination directory for nuscenes. 
    '''
    blobs_source_directory = os.environ.get('HOME',DATASET_DEFAULT_PATH)+DATASET_DEFAULT_PATH
    dataset_dest_directory = os.environ.get('HOME',DATASET_DEFAULT_PATH)+DATASET_DEFAULT_PATH
    dataset_dest_directory = os.path.join(dataset_dest_directory,'data/nuscenes')
    blobs_list = os.listdir(blobs_source_directory)
    blobs_file_prefix = 'v1.0'

    for i in range(len(blobs_list)):
        try:
            doc_name = blobs_list[i]
            if doc_name.startswith('v'):
                src_dir = os.path.join(blobs_source_directory,doc_name)
                print("moving files from %s to %s...\n\n"%(src_dir, dataset_dest_directory))
                num_files,num_transferred = moveFilesToNewPath(src_dir, dataset_dest_directory)
                print("moved %i out of %i files from %s\n\n"%(num_files,num_transferred,src_dir))
    
        except Exception as e:
            print("Error with %s"%(os.path.join(blobs_source_directory,doc_name)))

if __name__ == "__main__":
    print("In main. Running functions")
    source_file_path = os.environ.get('HOME',DATASET_DEFAULT_PATH)+ '/data/nuscenes/v1.0-test_blobs.tar'
    dest_path_1 = os.environ.get('HOME',DATASET_DEFAULT_PATH)+ '/data/nuscenes/v1.0-test_blobs'
    dest_path_2 = os.environ.get('HOME',DATASET_DEFAULT_PATH)+ DATASET_DEFAULT_PATH+'/v1.0-test_blobs'
    decodeFileToDirectory(source_file_path,dest_path_2)
