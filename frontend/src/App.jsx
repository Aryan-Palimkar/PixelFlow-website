import { useState } from 'react';
import DiffusionInfoModal from './components/DiffusionModal';
import DCGANInfoModal from './components/DCGANmodalInfo';
import api from './api';

function App() {
  const [selectedModel, setSelectedModel] = useState('dcgan');
  const [generatedImage, setGeneratedImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showDcganInfo, setShowDcganInfo] = useState(false);
  const [showDiffusionInfo, setShowDiffusionInfo] = useState(false);

  const toggleDiffusionInfo = () => setShowDiffusionInfo(false);
  const toggleDCGANInfo = () => setShowDcganInfo(false);

  const generateImage = async () => {
    setIsLoading(true);

    try {
      const response = await api.post('http://localhost:8000/generate', {
        model: selectedModel,
      });
      console.log('Response received:', response);
      setGeneratedImage(response.data.generated_image);
    } catch (error) {
      console.error('Error generating image:', error);
      alert('Failed to generate image. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 to-purple-900 text-white">
      <header className="pt-6 pb-4 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-pink-400 to-blue-400">
            C10 Pixel Flow
          </h1>
          <p className="mt-2 text-purple-200">Generate unique anime faces</p>
        </div>
      </header>

      <main className="max-w-4xl mx-auto p-4">
        <div className="bg-black bg-opacity-30 rounded-xl p-6 backdrop-blur-sm">
          <div className="flex flex-col md:flex-row gap-6">
            {/* Model Selection */}
            <div className="w-full md:w-1/3">
              <div className="mb-6">
                <h2 className="text-xl font-semibold mb-2">Model Selection</h2>
                <div className="flex gap-2 flex-wrap">
                  {/* DCGAN Button */}
                  <div className="flex items-center gap-2">
                    <button
                      className={`px-4 py-2 rounded-full transition ${
                        selectedModel === 'dcgan'
                          ? 'bg-pink-600 text-white'
                          : 'bg-gray-800 hover:bg-gray-700'
                      }`}
                      onClick={() => setSelectedModel('dcgan')}>
                      DCGAN
                    </button>
                    <button
                      className="p-[5px] rounded-full bg-pink-800 hover:bg-pink-700 transition flex items-center justify-center"
                      onClick={() => setShowDcganInfo(true)}
                      aria-label="Show DCGAN model information">
                      <i className="fa-solid fa-circle-info"></i>
                    </button>
                  </div>

                  {/* Diffusion Button */}
                  <div className="flex items-center gap-2">
                    <button
                      className={`px-4 py-2 rounded-full transition ${
                        selectedModel === 'diffusion'
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-800 hover:bg-gray-700'
                      }`}
                      onClick={() => setSelectedModel('diffusion')} >
                      Diffusion
                    </button>
                    <button
                      className="p-[6px] rounded-full bg-blue-800 hover:bg-blue-700 transition flex items-center justify-center"
                      onClick={() => setShowDiffusionInfo(true)}
                      aria-label="Show Diffusion model information" >
                      <i className="fa-solid fa-circle-info"></i>
                    </button>
                  </div>
                </div>
              </div>

              {/* Generation Section */}
              <div className="mb-6">
                <h2 className="text-xl font-semibold mb-2">Generation</h2>
                <button
                  className="w-full bg-gradient-to-r from-pink-500 to-blue-500 hover:from-pink-600 hover:to-blue-600 text-white font-medium py-3 px-4 rounded-lg transition flex items-center justify-center"
                  onClick={generateImage}
                  disabled={isLoading}>
                  {isLoading ? 'Generating...' : 'Generate New Face'}
                </button>
                <p className="text-sm text-center mt-2 text-gray-300">
                  Using {selectedModel === 'dcgan' ? 'DCGAN' : 'Diffusion'} model
                </p>
              </div>
            </div>

            {/* Generated Image Section */}
                  <div className="w-full md:w-2/3">
                    <div className="bg-gray-900 rounded-lg overflow-hidden h-64 md:h-80 flex items-center justify-center">
                    {isLoading ? (
                      <p>Generating anime face...</p>
                    ) : generatedImage ? (
                      <img
                      src={generatedImage}
                      alt="Generated anime face"
                      className="w-[256px] h-[256px] object-contain" />
                    ) : (
                      <p className="text-gray-400">No image generated yet</p>
                    )}
                    </div>
                    <p className="mt-4 text-center text-sm text-gray-300">
                    Click 'Generate New Face' to create unique anime characters
                    </p>
                  </div>
                  </div>
                </div>
                </main>

                {/* Modals */}
      {showDcganInfo && <DCGANInfoModal toggleDCGANInfoModal={toggleDCGANInfo} />}
      {showDiffusionInfo && <DiffusionInfoModal toggleDiffusionInfoModal={toggleDiffusionInfo} />}
    </div>
  );
}

export default App;
