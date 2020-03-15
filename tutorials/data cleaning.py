"""
Tutorial for analysing problems with the annotations.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import trainer.lib as lib
import trainer.ml as ml

if __name__ == '__main__':
    # Load dataset and split to be analysed
    ds_name, split_name = 'US_BONE', 'import'

    sess = lib.Session()

    ds: lib.Dataset = sess.query(lib.Dataset).filter(lib.Dataset.name == ds_name).first()
    split = ds.get_split_by_name(split_name)

    counter_all, counter_wrong = 0, 0
    for s in split.sbjts:
        print(s.name)
        for im_i, im in enumerate(s.ims):
            im_arr = im.get_ndarray()
            for gt in im.semseg_masks:
                gt_arr = gt.get_ndarray()
                counter_all += 1
                for mask_i in range(gt_arr.shape[2]):
                    tmp_mask = gt_arr[:, :, mask_i]
                    ss_cls_name = gt.tpl.ss_classes[mask_i].name
                    if not np.max(tmp_mask):
                        counter_wrong += 1
                        fig, (ax1, ax2) = plt.subplots(1, 2)
                        fig.suptitle(f'{s.name}: {im_i}, {gt.for_frame} {ss_cls_name}')
                        sns.heatmap(im_arr[gt.for_frame, :, :, 0], ax=ax1)
                        sns.heatmap(tmp_mask, ax=ax2)
                        ml.logger.save_fig(fig)
                        plt.close(fig)
                        # ml.logger.get_visboard().add_figure(fig)

    print(counter_all)
    print(counter_wrong)
