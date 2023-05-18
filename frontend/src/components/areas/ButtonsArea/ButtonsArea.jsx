import React from 'react'
import SaveButton from '../../buttons/SaveButton/SaveButton'
import ExportButton from '../../buttons/ExportButton/ExportButton'
import OpenButton from '../../buttons/OpenButton/OpenButton'
import ProjectTitle from '../../ProjectTitle/ProjectTitle'
import NewButton from '../../buttons/NewButton/NewButton'
import BuildButton from '../../buttons/BuildButton/BuildButton'
import './ButtonsArea.css'

export default function ButtonsArea() {
    return (
    <div className='buttons-area'>
        <ProjectTitle />
        <NewButton />
        <OpenButton />
        <SaveButton />
        <ExportButton />
        <BuildButton />
    </div>
    )
}
