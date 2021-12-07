using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.UI;

public class RunInferenceMobileNet : MonoBehaviour
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
    public Dropdown backendDropdown;
    
    private Model model;
    private IWorker engine;
    private Dictionary<string, Tensor> inputs = new Dictionary<string, Tensor>();
    private string[] labels;
    private RenderTexture targetRT;

    void Start()
    {
        Application.targetFrameRate = 60;
        Screen.orientation = ScreenOrientation.LandscapeLeft;
        AddBackendOptions();
        //parse neural net labels
        labels = labelsAsset.text.Split('\n');
        //load model
        model = ModelLoader.Load(srcModel);
        #if UNITY_WEBGL
            useGPU = false;
            backendDropdown.enabled = false;
        #endif
        //format input texture variable
        targetRT = RenderTexture.GetTemporary(inputResolutionX, inputResolutionY, 0, RenderTextureFormat.ARGBHalf);
        //execute inference
        SelectBackendAndExecuteML();
    }

    public void ExecuteML(int imageID)
    {
        #if UNITY_WEBGL
            useGPU = false;
        #endif
        selectedImage = imageID;
        displayImage.texture = inputImage[selectedImage];
        engine = WorkerFactory.CreateWorker(model, useGPU ? WorkerFactory.Device.GPU : WorkerFactory.Device.CPU);
        //preprocess image for input
        var input = new Tensor(PrepareTextureForInput(inputImage[selectedImage], !useGPU), 3);
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
        engine.Dispose();
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

    public void AddBackendOptions()
    {
        List<string> options = new List<string> ();
        options.Add("CSharpBurst");
        options.Add("ComputePrecompiled");
        backendDropdown.ClearOptions ();
        backendDropdown.AddOptions(options);
    }

    public void SelectBackendAndExecuteML()
    {
        if (backendDropdown.options[backendDropdown.value].text == "CSharpBurst")
        {
            useGPU = false;
        }
        
        if (backendDropdown.options[backendDropdown.value].text == "ComputePrecompiled")
        {
            useGPU = true;
        }
        ExecuteML(selectedImage);
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
