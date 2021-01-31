def visualize_detection(img_file, dets, classes=[], thresh=0.6):
        """
        visualize detections in one image
        Parameters:
        ----------
        img : numpy.array
            image, in bgr format
        dets : numpy.array
            ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
            each row is one object
        classes : tuple or list of str
            class names
        thresh : float
            score threshold
        """
        import random
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        img=mpimg.imread(img_file)
        plt.imshow(img)
        height = img.shape[0]
        width = img.shape[1]
        colors = dict()
        for det in dets:
            (klass, score, x0, y0, x1, y1) = det
            if score < thresh:
                continue
            cls_id = int(klass)
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            xmin = int(x0 * width)
            ymin = int(y0 * height)
            xmax = int(x1 * width)
            ymax = int(y1 * height)
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=3.5)
            plt.gca().add_patch(rect)
            class_name = str(cls_id)
            if classes and len(classes) > cls_id:
                class_name = classes[cls_id]
            plt.gca().text(xmin, ymin - 2,
                            '{:s} {:.3f}'.format(class_name, score),
                            bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                                    fontsize=12, color='white')
        plt.show()



def flip_bbox(bboxes):

    # bbox must be a list or array with values [cls_idx, xmin, ymin, xmax, ymax] * nBoxes
    # applies an horizontal flip to bboxes and returns the updated list

    num_attributes = 5
    num_bboxes = len(bboxes) // num_attributes

    flipped_bboxes = []

    for iBox in range(num_bboxes):
        this_bbox = bboxes[num_attributes*iBox:num_attributes*( 1+iBox )]

        flipped_bboxes.append( this_bbox[0] )

        # horizontal flip only
        original_xmin = this_bbox[1]
        original_xmax = this_bbox[3]
        new_xmin = 1 - original_xmax
        new_xmax = 1 - original_xmin

        flipped_bboxes.append(new_xmin)
        flipped_bboxes.append(this_bbox[2])
        flipped_bboxes.append(new_xmax)
        flipped_bboxes.append(this_bbox[4])

    return flipped_bboxes

