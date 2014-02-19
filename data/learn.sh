#!/bin/bash

set -e

BUILD="../build/";

${BUILD}learn \
	a/frame_001932.jpg a/1932_no_light_result.png a/ground_truth.png a/render_mask.png \
	bgsub2/view-0200-cam1.png bgsub2/view-0000-cam1.png bgsub2/ground_truth.png - \
	bgsub3/view-0300-cam2.png bgsub/view-0149-cam2.png bgsub3/ground_truth.png - \
	bgsub/view-0209-cam2.png bgsub/view-0149-cam2.png bgsub/ground_truth.png - \
	b/target.png b/0001_no_light_result.png b/ground_truth.png b/render_mask.png \
	c/small_IMG_1546.JPG c/no_light_result.png c/ground_truth.png c/render_mask.png \
	keys/s_IMG_1709.JPG keys/no_light_result.png keys/ground_truth.png keys/mask.png \
	hand1/background.png hand1/frame.png hand1/ground_truth.png - \
	hand2/background.png hand2/frame.png hand2/ground_truth.png - 
