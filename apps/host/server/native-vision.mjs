/** Prepared vision files stay in the source config; activation belongs to a loaded instance. */
export function runtimeVision(config,requested){
 const mode=requested??(config.ui_vision_enabled===false?'no':config.ui_vision_device);
 if(mode!==undefined&&!['yes','no','cpu','gpu'].includes(mode))throw new Error('Choose whether to enable picture reading.');
 if(mode==='no')return undefined;
 if(mode===undefined)return config.vision;
 if(!config.vision)throw new Error('Prepare this model’s vision files before enabling picture reading.');
 if(mode==='cpu'&&config.vision.cpu_exe){const {cpu_exe,...rest}=config.vision;return {...rest,exe:cpu_exe,gpu:false};} // AMD: the CPU reader program, not the HIP one with the GPU hidden
 return {...config.vision,...(mode==='cpu'?{gpu:false}:mode==='gpu'?{gpu:true}:{})};
}
