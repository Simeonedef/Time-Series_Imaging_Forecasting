import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

class visualization:

    
    def __init__(self,model):
        self.model = model
        
    def set_image(self,image):
        self.image = image    
        
    def get_layer_names(self,show=True):
        model = self.model
        
        #Get layer names of model
        layer_names=[]
        for layer in model.layers:
            layer_names.append(layer.name)
            
        #Plot    
        if (show):
            for i in range(len(layer_names)):
                print(layer_names[i],i)
        return layer_names
    
    def get_layer_shapes(self,show=True):
        model = self.model
        
        layer_output_shapes = []
        for layer in model.layers:
            layer_output_shapes.append(layer.output_shape)
        if (show):
            for i in range(len(layer_output_shapes)):
                print(layer_output_shapes[i],'layer: ',i)
        return layer_output_shapes
    
        
    def set_layer_index(self,layer_index):
        self.layer_index=layer_index
       
    def out_layer_vis(self,show=True):
        
        model = self.model
        image = self.image
        layer_index = self.layer_index
        
        layer_name = self.get_layer_names(show=False)[layer_index]
        
        output = [layer.output for layer in model.layers if layer.name in layer_name]

        # Create a connection between the input and those target outputs
        activations_model = tf.keras.models.Model(model.inputs, outputs=output)
        activations_model.compile(optimizer='adam', loss='mse')
        
        # Get their outputs
        activ = activations_model.predict(np.array([image]))
        
        N_filter = activ.shape[3]
        
        Ncol = 4
        Nrow = 2
        try:            
            if (show):
                fig, axis = plt.subplots(Nrow,Ncol)
                axis = axis.ravel()
                for i in range(N_filter):
                    axis[i].imshow(activ[0,:,:,i])
                    axis[i].axis('off')
                plt.tight_layout()
                plt.subplots_adjust(wspace=0.1, hspace=0.1)
                plt.show()   
                
                
        except:
            pass
        return activ

        
    def activity_map_vis(self,show=True):
        model = self.model
        image = self.image
        layer_index = self.layer_index
        
        layer_name = self.get_layer_names(show=False)[layer_index]
        
        # Create a graph that outputs target convolution and output
        grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
        
        pred_index=0 #how many points to forecast -> Constant
        
        # Get the score for target class
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(np.array([image])) #size:(1,img,img,filter) ; (1,1)
            loss = predictions[:, pred_index]
        
        # Extract filters and gradients
        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]
        
        # Average gradients spatially
        weights = tf.reduce_mean(grads, axis=(0, 1))
        
        # Build a ponderated map of filters according to gradients importance
        cam = np.ones(output.shape[0:2], dtype=np.float32)
        
        for index, w in enumerate(weights):
            cam += w * output[:, :, index]
        
        #Get size of filters
        filter_img_size = self.get_layer_shapes(show=False)[layer_index][2]
        # Heatmap visualization
        cam = cv2.resize(cam.numpy(), (filter_img_size, filter_img_size))
        cam = np.maximum(cam, 0)
        heatmap = (cam - cam.min()) / (cam.max() - cam.min())
        
        cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
        
        if show:
            fig, axis = plt.subplots(1,3)
            axis = axis.ravel()
            axis[0].imshow(image[:,:,0])
            axis[0].axis('off')            
            axis[1].imshow(image[:,:,1])
            axis[1].axis('off')            
            axis[2].imshow(cam)
            axis[2].axis('off')
            plt.tight_layout()
            plt.subplots_adjust(wspace=000.1, hspace=0.01)
            plt.show()
        return cam
