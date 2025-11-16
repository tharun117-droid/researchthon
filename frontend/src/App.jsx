//New: Plus

import { useRef, useState } from 'react'
import './App.css'
import InputText from './Components/InputText'
import OutputText from './Components/OutputText'
import InputDisplay from './Components/InputDisplay'

function App() {
  const divref= useRef()
  // Input storage
  const [text, setText] = useState(["hello from fe"])

  // Output storage -> to b done
  const [response, setResponse]= useState("")

  // Img storage -> to b done
  const [image, setImage]= useState(null)

  return (
    <div>
      <OutputText response= {response}></OutputText>
      <div>{text.map((text) => {return <InputDisplay text={text}></InputDisplay>})}</div>
      <InputText setText={setText} text={text} setImage={setImage}></InputText>
    </div>
  )
}

export default App
