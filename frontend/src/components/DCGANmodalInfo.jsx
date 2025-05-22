export default function DCGANInfoModal({ toggleDCGANInfoModal }) {
    return(

            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
              <div className="bg-gray-900 rounded-xl p-6 max-w-lg mx-4 border border-pink-500">
                <h2 className="text-2xl font-bold text-pink-400 mb-4">DCGAN Model Info</h2>
                
                <div className="space-y-4 text-gray-200">
                  <div>
                    <h3 className="text-lg font-semibold text-pink-400">About DCGAN</h3>
                    <p>DCGAN (Deep Convolutional Generative Adversarial Network) is a class of neural networks designed for generating realistic images by pitting two networks against each other.</p>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-semibold text-pink-400">Architecture</h3>
                    <p>Our DCGAN implementation uses:</p>
                    <ul className="list-disc pl-5 space-y-1 mt-2">
                      <li>Generator with transposed convolutional layers for image synthesis</li>
                      <li>Discriminator with convolutional layers for adversarial evaluation</li>
                      <li>Batch normalization for stable training</li>
                      <li>LeakyReLU activations for improved gradient flow</li>
                    </ul>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-semibold text-pink-400">Training Process</h3>
                    <p>DCGAN was relatively straightforward. There were a few instances of mode collapse, resulting in repetitive, distorted faces, but a few adjustments resolved those issues. The architecture was simple, and training didn't take too long. After addressing the challenges, we ended up with some decent(mostly cursed) images.</p>
                  </div>
                </div>
                
                <div className="mt-6 flex justify-end">
                  <button 
                    className="px-4 py-2 bg-pink-600 hover:bg-pink-700 rounded-lg transition"
                    onClick={() => toggleDCGANInfoModal()}
                  >
                    Close
                  </button>
                </div>
              </div>
            </div>
    );
}