import { useContext, useState } from "react"
import { MainContext } from "../../context/MainContext"
import ExportDrop from "../../areas/ExportDrop/ExportDrop";

export default function ExportButton() {
  const [modalActive, setModalActive] = useState(false)

  const onExport = () => {
    setModalActive(!modalActive)
  }

  return (
    <div>
      <button onClick={onExport}>toBlock</button>
      <ExportDrop mode = {modalActive}/>
    </div>
  )
}
