#!/bin/bash

set -e

INITIAL_DIR=$(pwd)
HERE=$(cd $(dirname $0) ; pwd)

BUILD="${HERE}/../build/";

cd ${HERE}/a && ${BUILD}emvisi2 1932_no_light_result.png frame_001932.jpg render_mask.png 
cd ${HERE}/bgsub2 && ${BUILD}emvisi2 view-0000-cam1.png view-0200-cam1.png 
cd ${HERE}/bgsub3 && ${BUILD}emvisi2 ../bgsub/view-0149-cam2.png view-0300-cam2.png 
cd ${HERE}/bgsub && ${BUILD}emvisi2 view-0149-cam2.png view-0209-cam2.png 
cd ${HERE}/b && ${BUILD}emvisi2 0001_no_light_result.png target.png render_mask.png 
cd ${HERE}/c && ${BUILD}emvisi2 no_light_result.png small_IMG_1546.JPG render_mask.png 
cd ${HERE}/keys && ${BUILD}emvisi2 no_light_result.png s_IMG_1709.JPG mask.png 
cd ${HERE}/hand1 && ${BUILD}emvisi2 background.png frame.png 
cd ${HERE}/hand2 && ${BUILD}emvisi2 background.png frame.png 

cd ${INITIAL_DIR}
echo Done.

