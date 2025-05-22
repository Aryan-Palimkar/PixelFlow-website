export default function DiffusionInfoModal({ toggleDiffusionInfoModal }) {
    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-gray-900 rounded-xl p-6 max-w-lg mx-4 border border-blue-500">
                <h2 className="text-2xl font-bold text-blue-400 mb-4">Diffusion Model Info</h2>
                
                <div className="space-y-4 text-gray-200">
                    <div>
                        <h3 className="text-lg font-semibold text-blue-400">About Diffusion Models</h3>
                        <p>Diffusion models are a class of generative models that learn to gradually denoise random noise patterns into coherent images through an iterative process.</p>
                    </div>
                    
                    <div>
                        <h3 className="text-lg font-semibold text-blue-400">Architecture</h3>
                        <p>Our diffusion implementation features:</p>
                        <ul className="list-disc pl-5 space-y-1 mt-2">
                            <li>U-Net backbone architecture with skip connections and residual blocks</li>
                            <li>Multi-Head Self Attention for improved coherence</li>
                            <li>Cosine beta scheduler for smoother noise transitions</li>
                            <li>Noise conditioning via timestep and noise-level embeddings</li>
                        </ul>
                    </div>
                    
                    <div>
                        <h3 className="text-lg font-semibold text-blue-400">Training Process</h3>
                        <p>Spent way too much time trying to understand the math(I still don’t really get it). In the initial attempt, model kept giving checkerboard patterns. Perceptual loss never worked, model gave cursed void faces everytime I tried it. I messed up a sampling parameter and got images that looked like amalgamation of warped faces.
                            After a lot of cursed images,
                            I finally ended up with results that were... okay. Guess that's a win.</p>
                    </div>
                </div>
                
                <div className="mt-6 flex justify-end">
                    <button 
                        className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition"
                        onClick={() => {
                            toggleDiffusionInfoModal();
                        }}
                    >
                        Close
                    </button>
                </div>
            </div>
        </div>
    );
}