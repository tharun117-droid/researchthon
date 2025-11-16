import { Plus } from "lucide-react";

function InputText({ setText, text, setImage }) {

  const handleFileSelect = (e) => {
    const file = e.target.files[0];

    if (file) {
      alert(`You selected: ${file.name}`);
      const url = URL.createObjectURL(file);
      setImage(url)
    // You can a dd custom logic here to display or upload the file
    }
  };

  return (
    <div className="input-container">
      <button
        className="plusbutton"
        title="Add Image/Video"
        onClick={() => document.getElementById("fileInput").click()}
      >
        <Plus size={20} />
      </button>
      <input
        id="fileInput"
        type="file"
        accept="image/*,video/*"
        style={{ display: "none" }}
        onChange={handleFileSelect}
      />
      <input
        title="inputTextBox"
        type="text"
        placeholder="Ask anything"
        className="textbox"
        onKeyDown={async (e) => {
          if (e.key === "Enter") {
            setText([...text, e.target.value]);
            e.target.value=""
          }
        }}
      />
    </div>
  );
}

export default InputText;
