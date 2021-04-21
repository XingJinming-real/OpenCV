"""
pixel read and write
    for a region of an array use slices for fast ops
    for a pixel use img.item(x,y,channel)
    or img.item(x,y,channel,value) to set its value
img properties
    img.shape :returns (w,h)
    img.size :returns num of pixels
split and merge img
    bChannel,gChannel,rChannel=cv.split(img)
    img=cv.merge((bChannel,gChannel,rChannel))
making borders for img
    cv.copyMakeBorder(img,top (num of pixels to be make in top position),
    bottom,left,right,BorderType)
    borderType - Flag defining what kind of border to be added. It can be following types:
        cv.BORDER_CONSTANT - Adds a constant colored border. The value should be given as next argument.
        cv.BORDER_REFLECT - Border will be mirror reflection of the border elements, like this : fedcba|abcdefgh|hgfedcb
        cv.BORDER_REFLECT_101 or cv.BORDER_DEFAULT - Same as above, but with a slight change, like this : gfedcb|abcdefgh|gfedcba
        cv.BORDER_REPLICATE - Last element is replicated throughout, like this: aaaaaa|abcdefgh|hhhhhhh
        cv.BORDER_WRAP - Can't explain, it will look like this : cdefgh|abcdefgh|abcdefg
    value - Color of border if border type is cv.BORDER_CONSTANT
"""
