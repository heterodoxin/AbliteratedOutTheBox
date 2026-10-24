"""Create the original Fieldnotes motion examples (Pillow + imageio-ffmpeg)."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math, subprocess
import imageio_ffmpeg
OUT=Path(__file__).resolve().parents[2]/'public/examples'
W,H=768,432
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',15)
small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',12)
serif=ImageFont.truetype('/System/Library/Fonts/Supplemental/Georgia.ttf',38)
def frame(t):
    im=Image.new('RGB',(W,H),'#f5f3eb');d=ImageDraw.Draw(im)
    d.text((40,30),'FIELDNOTES',font=font,fill='#224d3d')
    d.text((40,112),'A little room\nfor your next idea.',font=serif,fill='#243e34',spacing=5)
    d.text((42,230),'Capture. Connect. Create.',font=font,fill='#647368')
    d.rounded_rectangle((42,285,202,324),20,fill='#224d3d');d.text((63,296),'Make a fresh start',font=small,fill='#ffffff')
    for i in range(3):
        y=95+i*36+int(math.sin(t*2*math.pi+i*.55)*10)
        x=444+i*16
        d.rounded_rectangle((x,y,x+196,y+175),18,fill=['#d5dfcf','#b9cdb9','#fffcf4'][i],outline='#b2c4b3',width=1)
        if i==2:
            d.ellipse((x+24,y+25,x+36,y+37),fill='#408260')
            d.text((x+24,y+55),'An idea worth keeping',font=small,fill='#243e34')
            for j,length in enumerate([135,113,128]):d.rounded_rectangle((x+24,y+88+j*17,x+length,y+93+j*17),2,fill='#d5dfcf')
    d.line((40,375,728,375),fill='#d5dfcf',width=1)
    d.text((40,390),'A SMALL SPACE FOR WHAT COMES NEXT',font=small,fill='#647368')
    for i in range(3):
        d.ellipse((672+i*18,392,678+i*18,398),fill='#224d3d' if int(t*3)%3==i else '#c4d1bd')
    return im
frames=[frame(i/72) for i in range(72)]
frames[0].save(OUT/'fieldnotes-motion-poster.png')
writer=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-vcodec','rawvideo','-s',f'{W}x{H}','-pix_fmt','rgb24','-r','18','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'fieldnotes-motion.mp4')],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
for im in frames:writer.stdin.write(im.tobytes())
writer.stdin.close()
if writer.wait():raise RuntimeError('Video encoding failed')
# A 4-second, 9 fps looping GIF with a separate still for pause/reduced motion.
gif_frames=[im.resize((576,324)).quantize(colors=96) for im in frames[::2]]
gif_frames[0].save(OUT/'fieldnotes-ideas.gif',save_all=True,append_images=gif_frames[1:],duration=111,loop=0,optimize=False,disposal=2)
print('Created MP4, animated GIF and still poster.')
