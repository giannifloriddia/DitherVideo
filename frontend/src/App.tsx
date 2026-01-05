import * as React from "react";
import NavBar from "./components/NavBar.tsx";
import SubmitForm from "./components/SubmitForm.tsx"; // Assuming .tsx extension is resolved by build tool

function App() {

    const [resultMedia, setResultMedia] = React.useState<string | null>(null);
    const [isVideo, setIsVideo] = React.useState<boolean>(false);

    const handleMediaSuccess = (mediaUrl: string, isVideoFile: boolean) => {
        setResultMedia(mediaUrl);
        setIsVideo(isVideoFile);
    }

    return (
        <>
            <NavBar />
            <div>
                <h1>Dither Lab</h1>
                <h2>Insert image or video</h2>

                <SubmitForm onMediaProcessed={handleMediaSuccess}></SubmitForm>

                {resultMedia && (
                    <div>
                        <h3>Result:</h3>
                        {isVideo ? (
                            <video
                                src={resultMedia}
                                controls
                                style={{ maxWidth: "100%", border: "1px solid #ccc" }}
                            />
                        ) : (
                            <img
                                src={resultMedia}
                                alt="Dithered Result"
                                style={{ maxWidth: "100%", border: "1px solid #ccc" }}
                            />
                        )}
                        <br />
                        <a href={resultMedia} download={isVideo ? "dithered_video.mp4" : "dithered_image.png"}>
                            Download {isVideo ? "Video" : "Image"}
                        </a>
                    </div>
                )}
            </div>
        </>
    );
}

export default App;