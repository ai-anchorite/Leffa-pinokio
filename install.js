module.exports = {
  run: [
  
    {
     when: "{{gpu !== 'nvidia'}}",
     method: "notify",
     params: {
       html: "This app requires an NVIDIA GPU."
     }, 
       next: null
    },

    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",               
          path: "app",                
        }
      }
    },
    
    {
      method: "shell.run",
      params: {
        venv: "env",               
        path: "app",               
        message: [
          "uv pip install gradio devicetorch",
          "uv pip install -r requirements.txt"
        ]
      }
    }
  ]
}
