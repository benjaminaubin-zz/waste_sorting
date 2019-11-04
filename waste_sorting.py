from Libraries.import_libraries import *
from Functions.functions_save_load import save_object, load_object

parser = argparse.ArgumentParser(description='Waste sorter launcher')
parser.add_argument('--mode', type=str, help='mode: training or testing',
                    default='training', choices=['training', 'testing'])
args = parser.parse_args()

## References ##
# https://towardsdatascience.com/how-to-build-an-image-classifier-for-waste-sorting-6d11d3c9c478
# http://cs230.stanford.edu/projects_spring_2019/reports/18678247.pdf
# https://colab.research.google.com/drive/18AN2AUM5sEsTMGUzFUL0FLSULtXF4Ps0#scrollTo=jdFgkFl3ztr7
# https://medium.com/udacity-pytorch-challengers/ideas-on-how-to-fine-tune-a-pre-trained-model-in-pytorch-184c47185a20
# https://towardsdatascience.com/deep-learning-performance-cheat-sheet-21374b9c4f45


class Waste_Sorter():
    def __init__(self):
        ## Device ##
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

        ## Loss ##
        self.loss = nn.CrossEntropyLoss()

        ## Parameters ##
        self.best_settings = {'optimize': 'adam', 'batch_size': 128,
                              'num_epochs': 30, 'lr': 6e-4, 'wd': 1e-5, 'step_size': 20, 'gamma': 0.1}
        self.choice_optim = 'adam'
        self.batch_size = 128
        self.num_epochs = 30
        self.lr = 6e-4
        self.wd = 1e-5
        self.step_size = 20
        self.gamma = 0.1
        self.name_model = f'optim_{self.choice_optim}_bs_{self.batch_size}_epoch_{self.num_epochs}_lr_{self.lr}_wd_{self.wd}_ss_{self.step_size}_gamma_{self.gamma}'
        print(self.name_model)

        ## Class ##
        self.class_names = ['cardboard', 'glass',
                            'metal', 'paper', 'plastic', 'trash']

        self.trashes = {'cardboard': 'yellow', 'glass': 'white', 'metal': 'yellow',
                        'paper': 'yellow', 'plastic': 'yellow', 'trash': 'yellow'}

    def load_data(self):
        data_dir = 'Data/'
        data_transforms = self.build_transformations()

        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                  data_transforms[x])
                          for x in ['train', 'test', 'test']}
        self.dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=self.batch_size,
                                                           shuffle=True, num_workers=8)
                            for x in ['train', 'test', 'test']}
        self.dataset_sizes = {x: len(image_datasets[x]) for x in [
            'train', 'test', 'test']}
        self.class_names = image_datasets['train'].classes
        self.num_classes = len(self.class_names)
        print(self.class_names)
        #self.class_names_imagenet = load_object('data/imagenet_classes.pkl')

    def build_transformations(self):

        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225])
            ]),
        }
        # data_transforms = {
        #     # Train uses data augmentation
        #     'train':
        #     transforms.Compose([
        #         transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        #         transforms.RandomRotation(degrees=15),
        #         transforms.ColorJitter(),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.CenterCrop(size=224),  # Image net standards
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406],
        #                              [0.229, 0.224, 0.225])  # Imagenet standards
        #     ]),
        #     # Validation does not use augmentation
        #     'valid':
        #     transforms.Compose([
        #         transforms.Resize(size=256),
        #         transforms.CenterCrop(size=224),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [
        #                              0.229, 0.224, 0.225])
        #     ]),
        # }

        return data_transforms

    def training(self, model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    inputs, labels = inputs.to(
                        self.device), labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            scheduler.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / \
                    self.dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)

        # Compute test error
        model.eval()
        test_loss = 0.0
        test_corrects = 0
        for inputs, labels in self.dataloaders['test']:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # statistics
            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data)

        test_loss = test_loss / self.dataset_sizes['test']
        test_acc = test_corrects.double() / self.dataset_sizes['test']

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            'test', test_loss, test_acc))

        return model

    def load_model(self):
        model = models.resnet101(pretrained=True)
        # Freeze model weights
        for param in model.parameters():
            param.requires_grad = True

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.num_classes)

        # for name, child in model.named_children():
        #     print(name)

        for name, child in model.named_children():
            if name in ['fc', 'layer4', 'avgpool']:
                print(name + ' is unfrozen')
                for param in child.parameters():
                    param.requires_grad = True
            else:
                print(name + ' is frozen')
                for param in child.parameters():
                    param.requires_grad = False

        # print(model)

        self.model = model.to(self.device)

    def train_model(self):
        # Optimizer
        print(f'Optimizer: {self.choice_optim}')

        if self.choice_optim == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=self.lr, momentum=0.9)

        elif self.choice_optim == 'rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr)

        elif self.choice_optim == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.wd)

        # lr scheduler
        self.exp_lr_scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=self.step_size, gamma=self.gamma)

        model_trained = self.training(
            self.model, self.loss, self.optimizer, self.exp_lr_scheduler, num_epochs=self.num_epochs)
        self.model_trained = model_trained
        torch.save(model_trained, f'Model/{self.name_model}.pkl')

    def load_trained_model(self):
        self.choice_optim = 'adam'
        self.batch_size = 128
        self.num_epochs = 30
        self.lr = 6e-4
        self.wd = 1e-5
        self.step_size = 20
        self.gamma = 0.1
        name = f'optim_{self.choice_optim}_bs_{self.batch_size}_epoch_{self.num_epochs}_lr_{self.lr}_wd_{self.wd}_ss_{self.step_size}_gamma_{self.gamma}'
        self.model = torch.load(f'Model/{name}.pkl')

    def predict(self, img_path):
        img = Image.open(img_path)
        data_transforms = self.build_transformations()
        input = data_transforms['test'](img)
        input.unsqueeze_(0)  # Add batch dimension
        input = input.to(self.device)
        output = self.model(input)
        _, pred = torch.max(output, 1)
        sm = torch.nn.Softmax()
        probas = sm(output).tolist()[0]
        print(
            f'img:{img_path} pred:{self.class_names[pred]} precision:{probas[pred]:.2f}')

        ## Annex ##

    def predict_real(self):
        data_dir = 'Data/real/'
        file_names = [fn for fn in os.listdir(data_dir)
                      if any(fn.endswith(ext) for ext in ['jpg'])]
        print(file_names)
        for img in file_names:
            self.predict(data_dir + img)

    def imshow(self, inputs, title=None):
        """Imshow for Tensor."""
        inputs = inputs.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inputs = std * inputs + mean
        inputs = np.clip(inputs, 0, 1)
        plt.imshow(inputs)
        if title is not None:
            plt.title(title)
        plt.show(block=False)
        input('Please press to close')
        plt.close()

    def plot_sample(self):
        inputs, classes = next(iter(self.dataloaders['train']))
        outputs = torchvision.utils.make_grid(inputs)
        self.imshow(outputs, title=[self.class_names[x] for x in classes])

    def visualize_model(self, model, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
        #fig = plt.figure()
        TP, TN, n_samples = 0, 0, 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.dataloaders['test']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                sm = torch.nn.Softmax()
                probas = sm(outputs)
                _, preds = torch.max(outputs, 1)
                np.set_printoptions(precision=1)
                for j in range(inputs.size()[0]):
                    n_samples += 1
                    id_pred = preds[j].tolist()
                    id_true = labels[j].tolist()
                    proba = probas[j].tolist()
                    print(
                        f'predicted: {self.class_names[id_pred]} true: {self.class_names[id_true]} Acc: {proba[id_pred]}')
                    print(f'probas: {np.array(proba)}')
                    print(f'class: {self.class_names}')

                    if id_true == id_pred:
                        TP += 1
                    else:
                        TN += 1
            input('Press..')

            print(f'TP: {TP/n_samples * 100} TN: {TN/n_samples * 100}')

            # print(labels, preds)
            # for j in range(inputs.size()[0]):
            #     images_so_far += 1
            #     ax = plt.subplot(num_images//2, 2, images_so_far)
            #     ax.axis('off')

            #     id = preds[j].tolist()
            #     id_ = labels[j].tolist()

            #     ax.set_title(
            #         f'predicted: {self.class_names[id]} true:{self.class_names[id_]}')
            #     self.imshow(inputs.cpu().data[j])

            #     if images_so_far == num_images:
            #         model.train(mode=was_training)
            #         return
            model.train(mode=was_training)

    def main(self):
        if args.mode == 'training':
            self.load_data()
            self.load_model()
            self.train_model()
        elif args.mode == 'testing':
            self.load_trained_model()
            # self.visualize_model(self.model)
            self.predict_real()


if __name__ == '__main__':
    waste_sorter = Waste_Sorter()
    waste_sorter.main()
