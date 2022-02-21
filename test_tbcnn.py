import tqdm as tqdm
from torch.utils.data import Dataloader
from sklearn.metrics import classification_report, confusion_matrix

print('Starting to test.....')
    # testing procedure
    pred = []
    true = []
    test_acc = 0.0
    total_loss = 0.0
    total = 0.0
    i = 0
    model = best_model
    while i < len(test_data):
        batch = get_batch(test_data, i, BATCH_SIZE)
        i += BATCH_SIZE
        test_inputs, test_labels = batch
        if USE_GPU:
           test_inputs, test_labels = test_inputs, test_labels.cuda()

        model.zero_grad()
        model.batch_size = len(test_labels)
        model.hidden = model.init_hidden()
        output = model(test_inputs)

        loss = loss_function(output, Variable(test_labels))

        _, pred = torch.max(output.data, 1)
        pred = np.asarray(pred, dtype='float')
        pred /= np.max(np.abs(pred))
        t = test_inputs.cpu().unique().numel()
        test_acc += (pred == test_labels).sum()
        total += len(test_labels)
        total_loss += loss.item() * len(test_inputs)
    print(['Testing results(Acc): %.3f' %(test_acc.item() / total)])    

