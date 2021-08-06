using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.UI;

public class RunInference : MonoBehaviour
{
    public Texture2D[] inputImage;
    public int selectedImage = 0;
    public int inputResolutionY = 224;
    public int inputResolutionX = 224;
    public RawImage displayImage;
    public NNModel srcModel;
    public TextAsset labelsAsset;
    public Text resultClassText;
    public Material preprocessMaterial;
    public bool useGPU = true;
    
    private Model model;
    private IWorker engine;
    private Dictionary<string, Tensor> inputs = new Dictionary<string, Tensor>();
    private string[] labels;
    private RenderTexture targetRT;

    void Start()
    {
        Application.targetFrameRate = 60;
        //parse neural net labels
        labels = labelsAsset.text.Split('\n');
        //load model and worker
        model = ModelLoader.Load(srcModel, false);
        #if UNITY_WEBGL
            useGPU = false;
        #endif
        engine = WorkerFactory.CreateWorker(model, useGPU ? WorkerFactory.Device.GPU : WorkerFactory.Device.CPU);
        //format input texture variable
        targetRT = RenderTexture.GetTemporary(inputResolutionX, inputResolutionY, 0, RenderTextureFormat.ARGBHalf);
        //execute inference
        ExecuteML(0);
    }

    public void ExecuteML(int imageID)
    {
        selectedImage = imageID;
        displayImage.texture = inputImage[selectedImage];
        //preprocess image for input
        #if UNITY_WEBGL
            var inputTexture = PrepareTextureForInput(inputImage[selectedImage], !useGPU);
            var input = new Tensor(1, inputResolutionX, inputResolutionY, 3);
            input = TextureToTensor(input, (Texture2D) inputTexture);
        #else
            var input = new Tensor(PrepareTextureForInput(inputImage[selectedImage], !useGPU), 3);
        #endif
        //execute neural net
        engine.Execute(input);
        //read output tensor
        var output = engine.PeekOutput();
        //select the best output class and print the results
        var res = output.ArgMax()[0];
        var label = labels[res];
        var accuracy = output[res];
        resultClassText.text = $"{label} {Math.Round(accuracy*100, 1)}﹪";
        //clean memory
        input.Dispose();
        Resources.UnloadUnusedAssets();
    }

    Texture PrepareTextureForInput(Texture2D src, bool needsCPUcopy)
    {
        RenderTexture.active = targetRT;
        //normalization is applied in the NormalizeInput shader
        Graphics.Blit(src, targetRT, preprocessMaterial);

        if (!needsCPUcopy)
            return targetRT; 
		
        var  result = new Texture2D(targetRT.width, targetRT.height, TextureFormat.RGBAHalf, false);
        result.ReadPixels(new Rect(0,0, targetRT.width, targetRT.height), 0, 0);
        result.Apply();
        return result;
    }

    Tensor TextureToTensor(Tensor input, Texture2D inputTexture)
    {
        var pixelData = inputTexture.GetPixels();
        for (var y = 0; y < inputResolutionY; y++)
        {
            for (var x = 0; x < inputResolutionX; x++)
            {
                input[0, inputResolutionY - 1 - y, x, 0] = pixelData[y * inputResolutionY + x][0];
                input[0, inputResolutionY - 1 - y, x, 1] = pixelData[y * inputResolutionY + x][1];
                input[0, inputResolutionY - 1 - y, x, 2] = pixelData[y * inputResolutionY + x][2];
            }
        }
        return input;
    }
    
    private void OnDestroy()
    {
        engine?.Dispose();

        foreach (var key in inputs.Keys)
        {
            inputs[key].Dispose();
        }
		
        inputs.Clear();
    }
}
