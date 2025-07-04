def squeeze2img(tensor):
    # return tensor.squeeze()[[15,8,3]]
    return tensor.squeeze()[[20]]

def squeeze2img_rgb(tensor):
    # return tensor.squeeze()[[15,8,3]]
    return tensor.squeeze()

def squeeze2img_chikusei(tensor):
    return tensor.squeeze()[[69,99,35]]
    # return tensor.squeeze()


def listmap(func, *iterables):
    return list(map(func, *iterables))


def cal_metric(func, output, target):
    if len(output.shape) == 5:
        output = output.squeeze(1)
        target = target.squeeze(1)
    return func(output, target)
