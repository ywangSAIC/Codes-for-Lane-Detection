import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import cv2
import utils.transforms as tf
import numpy as np
import models
#from models import sync_bn
import dataset as ds
from options.options import parser
import torch.nn.functional as F

best_mIoU = 0
channel_color = [[0],[1],[2],[0,1]]
colors = ((255,0,0),(0,255,0), (0,0,255), (255,255,0))
scale = []
#gamma = 1.0
#inv_gamma = 1/gamma
#tabel = np.array( [(( i /255.0) ** inv_gamma) * 255  for i in range(256)]).astype('uint8')

#debug
import pdb

def main():
    global args, best_mIoU, scale
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu) for gpu in args.gpus)
    args.gpus = len(args.gpus)

    scale = [(1920.0/ args.img_width), (1080-440.0)/args.img_height]
    #if args.no_partialbn:
    #    sync_bn.Synchronize.init(args.gpus)

    if args.dataset == 'VOCAug' or args.dataset == 'VOC2012' or args.dataset == 'COCO':
        num_class = 21
        ignore_label = 255
        scale_series = [10, 20, 30, 60]
    elif args.dataset == 'Cityscapes':
        num_class = 19
        ignore_label = 255 
        scale_series = [15, 30, 45, 90]
    elif args.dataset == 'ApolloScape':
        num_class = 37 
        ignore_label = 255 
    elif args.dataset == 'CULane':
        num_class = 5
        ignore_label = 255
    else:
        raise ValueError('Unknown dataset ' + args.dataset)

    model = models.ERFNet(num_class, partial_bn=not args.no_partialbn)
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    model = torch.nn.DataParallel(model, device_ids=range(args.gpus)).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mIoU = checkpoint['best_mIoU']
            torch.nn.Module.load_state_dict(model, checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))


    cudnn.benchmark = True
    cudnn.fastest = True

    # Data loading code

    test_loader = torch.utils.data.DataLoader(
        getattr(ds, args.dataset.replace("CULane", "VOCAug") + 'DataSet')(dataset_path = args.dataset_path, data_list=args.val_list, transform=torchvision.transforms.Compose([
            tf.GroupROICrop(ROI=(0, 200, 1920, 640)),
            tf.GroupRandomScaleNew(size=(args.img_width, args.img_height), interpolation=[cv2.INTER_LINEAR]),
            tf.GroupNormalize(mean=(input_mean, (0, )), std=(input_std, (1, ))),
        ])), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    
    # define loss function (criterion) optimizer and evaluator
    weights = [1.0 for _ in range(5)]
    weights[0] = 0.4
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = torch.nn.NLLLoss(ignore_index=ignore_label, weight=class_weights).cuda()
    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
    optimizer = torch.optim.SGD(policies, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    evaluator = EvalSegmentation(num_class, ignore_label)

    ### evaluate ###
    validate(test_loader, model, criterion, 0, evaluator)
    return



def uv2xyz(u,v):
    fx = 1994.5
    cx = 946.3
    fy = 1994.7
    cy = 514.8
    y = 2.0 
    rot = np.array([[ 0.03795, -0.99918, 0.0140702], [0.0042642, -0.0139182, -0.99989], [0.99927, 0.0380123, 0.0037325]])
    trans = np.array([1.879, -0.006, 2.010]).T
    #trans = np.array([-0.106, 2.002, -1.886]).T
    rot_inv = np.linalg.inv(rot)
    M = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    M_inv = np.linalg.inv(M)
    p = np.array([u,v,1])
    X = np.dot(M_inv,p)
    X = X/X[1] * y
    #X = np.dot(rot_inv, X)  + trans
    return X[:]

def get_pix_base_link(p_ground):
    x = -p_ground[1] * 10 + 60
    y = 500 - p_ground[0] * 10 
    return (int(x),int(y))

def get_pix(p_ground):
    x = p_ground[0] * 10 + 60
    y = 500 - p_ground[2] * 10 
    return (int(x),int(y))
def validate(val_loader, model, criterion, iter, evaluator, logger=None):

    batch_time = AverageMeter()
    losses = AverageMeter()
    IoU = AverageMeter()
    mIoU = 0

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input,  img_name) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input, volatile=True)
        # compute output
        output, output_exist = model(input_var)

        # measure accuracy and record loss

        output = F.softmax(output, dim=1)

        pred = output.data.cpu().numpy() # BxCxHxW
        pred_exist = output_exist.data.cpu().numpy() # BxO

        for cnt in range(len(img_name)):
            directory = 'predicts/vgg_SCNN_DULR_w9' + img_name[cnt][:-10]
            if not os.path.exists(directory):
                os.makedirs(directory)
            file_exist = open('predicts/vgg_SCNN_DULR_w9'+img_name[cnt].replace('.jpg', '.exist.txt'), 'w')
            src_img_1 = cv2.imread(args.dataset_path+"/.."+ img_name[cnt])
            src_img = input_var[cnt].permute(1,2,0).data.cpu().numpy().copy()
            local_map = np.zeros((500, 120,3), np.uint8)
            local_map[:] = 255
            cv2.line(local_map, (60,0),(60,500),(0,0,0),2)
            #cv2.imwrite('predicts/vgg_SCNN_DULR_w9'+img_name[cnt].replace('.jpg', '_src.jpg'),(input_var[cnt].permute(1,2,0).data.cpu().numpy()) )
            #cv2.imwrite('predicts/vgg_SCNN_DULR_w9'+img_name[cnt].replace('.jpg', '_src.png'),(pred[cnt][0].reshape((208,976)) * 64).astype('uint8') )
            for num in range(4):
                if pred_exist[cnt][num] > 0.5:
                    file_exist.write('1 :\n')
                    prob_map = (pred[cnt][num+1]*255).astype('uint8')
                    ret, thresh = cv2.threshold(prob_map,150,255,0)
                    _, contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    #cv2.drawContours(src_img, contours,-1,colors[num],1)

                    x = []
                    y = []
                    for c in contours:
                        if c.shape[0] >3:
                            for pt in c:
                                x.append(pt[0][0])
                                y.append(pt[0][1])

                    if len(x) > 10:  
                        z = np.polyfit(y,x,1)
                        p = np.poly1d(z)
                        start_idx = 50
                        cv2.line(src_img, (int(p(start_idx)),start_idx) , (int(p(args.img_height)), args.img_height), colors[num],3)
                        
                        p0 = (int(p(start_idx) * scale[0]), int(start_idx * scale[1] + 440))
                        p1 =  (int(p(args.img_height) * scale[0]), int(args.img_height * scale[1]) + 440)

                        cv2.line(src_img_1, p0 , p1, colors[num],3)

                        file_exist.write("line in ori_img: p1: ("+ str(p0[0])+", "+str(p0[1] )+ "), p2: ("+str(p1[0])+", "+str(p1[1])+")\n")
                        p0_ground = uv2xyz(*p0)
                        p1_ground  = uv2xyz(*p1)


                        cv2.line(local_map, get_pix(p0_ground), get_pix(p1_ground), colors[num],4)
                        file_exist.write("line on ground(camera): p1: ("+ str(p0_ground[0])+", "+ str(p0_ground[1])+", "+str(p0_ground[2])+ "), p2: ("+str(p1_ground[0])+", "+str(p1_ground[1])+", "+str(p1_ground[2])+")\n")
                    file_exist.write("\n")    
                        # z = np.polyfit(x,y,2)
                        # p = np.poly1d(z)
                        # polys = np.empty((args.img_width,1,2), np.int32)
                        # for i in range(args.img_width):
                        #     polys[i][0][0] = i
                        #     polys[i][0][1] = int(p(i))
                        # #pdb.set_trace()            
                        # cv2.polylines(src_img,[polys], False, colors[num],3)
                    #cv2.imwrite('predicts/vgg_SCNN_DULR_w9'+img_name[cnt].replace('.jpg', '_'+str(num+1)+'_avg.png'), thresh)
                    #try:
                    #    vx,vy,x,y = cv2.fitLine(contours[0],cv2.DIST_L2, 0,0.01,0.01)
                    #    print("fited")
                    #except Exception as e:
                    #    print(e)
                    #    continue
                    #for c in channel_color[num]:
                    #    src_img[:,:,c] += prob_map.astype('uint8')
    
                else:
                    file_exist.write('0 \n')

            src_img = cv2.normalize(src_img,None,0,255,cv2.NORM_MINMAX, cv2.CV_8U)
            #src_img =cv2.LUT(src_img.astype('uint8'), tabel)
            cv2.imwrite('predicts/vgg_SCNN_DULR_w9'+img_name[cnt].replace('.jpg', '_ret.png'), src_img)
            cv2.imwrite('predicts/vgg_SCNN_DULR_w9'+img_name[cnt].replace('.jpg', '_ret_ori.png'), src_img_1)
            cv2.imwrite('predicts/vgg_SCNN_DULR_w9'+img_name[cnt].replace('.jpg', '_projected.png'), local_map)
            file_exist.close()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time)))

    print('finished, #test:{}'.format(i) )

    return mIoU


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def update(self, val, n=1):
        if self.val is None:
            self.val = val
            self.sum = val * n
            self.count = n
            self.avg = self.sum / self.count
        else:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


class EvalSegmentation(object):
    def __init__(self, num_class, ignore_label=None):
        self.num_class = num_class
        self.ignore_label = ignore_label

    def __call__(self, pred, gt):
        assert (pred.shape == gt.shape)
        gt = gt.flatten().astype(int)
        pred = pred.flatten().astype(int)
        locs = (gt != self.ignore_label)
        sumim = gt + pred * self.num_class
        hs = np.bincount(sumim[locs], minlength=self.num_class**2).reshape(self.num_class, self.num_class)
        return hs


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # decay = 0.1**(sum(epoch >= np.array(lr_steps)))
    decay = ((1 - float(epoch) / args.epochs)**(0.9))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


if __name__ == '__main__':
    main()
