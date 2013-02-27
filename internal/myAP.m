function [ap] = myAP(confidence, gt, cls)
assert(length(confidence)==length(gt));

cls_idx = (gt==cls);
gt(cls_idx) = 1;
gt(~cls_idx) = -1;

[so,si]=sort(-confidence);
tp=gt(si)>0;
fp=gt(si)<0;

fp=cumsum(fp);
tp=cumsum(tp);
rec=tp/sum(gt>0);
prec=tp./(fp+tp);

ap=VOCap(rec,prec);