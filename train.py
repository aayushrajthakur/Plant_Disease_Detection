import tensorflow as tf
from model import create_model
from data_preprocessing import load_data

def train_model(test, epochs=10, batch_size=32):
    images, labels = load_data(test)
    model = create_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(images, labels, epochs=epochs, batch_size=batch_size)
    model.save('plant_disease_model.keras') 

if __name__ == "__main__":
    train_model('test')
