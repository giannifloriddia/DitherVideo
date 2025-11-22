import * as React from "react";
import NavBar from "./components/NavBar.tsx";
import SubmitForm from "./components/SubmitForm.tsx"; // Assuming .tsx extension is resolved by build tool

function App() {

    const [resultImage, setResultImage] = React.useState<string | null>(null);

    const handleImageSuccess = (imageUrl: string) => {
        setResultImage(imageUrl);
    }

    return (
        <>
            <NavBar />
            <div>
                <h1>Dither Lab</h1>
                <h2>Insert image</h2>

                <SubmitForm onImageProcessed={handleImageSuccess}></SubmitForm>

                {resultImage && (
                    <div>
                        <h3>Result:</h3>
                        <img
                            src={resultImage}
                            alt="Dithered Result"
                            style={{ maxWidth: "100%", border: "1px solid #ccc" }}
                        />
                        <br />
                        <a href={resultImage} download="dithered_image.png">Download Image</a>
                    </div>
                )}
            </div>
        </>
    );
}

export default App;