ffmpeg -framerate 12 -i /scratchdata/alcove/combined/%d.png -c:v libx264 -r 30 -pix_fmt yuv420p output.mp4

