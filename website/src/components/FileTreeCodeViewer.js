import React, { useState, useRef, useEffect } from 'react';
import FileTree from './FileTree';
import CodeBlock from '@theme/CodeBlock';


export default function FileTreeCodeViewer({
    template,
    treeId,
    viewState,
    defaultItem = {
        data: {
            filePath: 'loading...',
            fileExtension: '',
            content: '',
        }
    }
}) {
    const environment = useRef();
    const [item, setItem] = useState(defaultItem);

    // simulate clicking the default value.
    useEffect(() => {
        try {
            const treeElement = environment.current;
            const treeEnvironmentContext = treeElement.treeEnvironmentContext;
            const viewState = treeEnvironmentContext.viewState;
            const viewStateTree = viewState[treeId];
            const selectedItems = viewStateTree.selectedItems;

            for (const selectedItem of selectedItems) {
                treeElement.invokePrimaryAction(selectedItem, treeId)
            }
        } catch (error) {
            return; // fail softly
        }
    });

    return (
        <div className="row">
            <div className="col col--4">
                <FileTree
                    template={template}
                    treeId={treeId}
                    viewState={viewState}
                    onPrimaryAction={setItem}
                    environment={environment}
                />
            </div>
            <div className="col col--8">
                <CodeBlock
                    className={`language-${item.data.fileExtension}`}
                    title={item.data.filePath}
                >
                    {item.data.content}
                </CodeBlock>
            </div>
        </div>
    )
}