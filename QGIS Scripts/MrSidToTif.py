import gdal
import os
import argparse

def MrSidToTif(in_dir, removeInput=False):
    for filename in os.listdir(in_dir):
        if filename.endswith('.sid'):
            in_filepath = os.path.join(in_dir, filename)
            gdal_file = gdal.Open(in_filepath)
            out_filepath = in_filepath.split('.')[0] + '.tif'
            gdal.Translate(out_filepath, gdal_file)
            if removeInput:
                os.remove(in_filepath)
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_file', help='The directory to process .sid files.')
    parser.add_argument('--removeInput', default=False, help='Boolean to delete input files after transform. Default False.')
    
    args = parser.parse_args()
    MrSidToTif(args.in_file, args.removeInput)