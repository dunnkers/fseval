import React from 'react';
import {
    UncontrolledTreeEnvironment,
    Tree,
    StaticTreeDataProvider
} from 'react-complex-tree';
import 'react-complex-tree/lib/style.css';

const readTemplate = (template, data = { items: {} }, directory = '') => {
    for (const [key, value] of Object.entries(template)) {
        // Passed objects should be strings, i.e. the raw file contents. This determines
        // whether a node is a file or a directory.
        const isDirectory = typeof value !== 'string'

        // Filepath. Because we started at `/root`, we should remove that.
        const filePath = `${directory}/${key}`.replace('/root', '')
        const fileExtension = filePath.split('.').pop();

        data.items[key] = {
            index: key,
            hasChildren: isDirectory,
            children: isDirectory ? Object.keys(value) : undefined,
            canMove: false,
            canRename: false,
            data: {
                filePath,
                fileExtension,
                key, // filename or directory name
                content: isDirectory ? null : value,
            },
        };

        if (isDirectory) {
            readTemplate(value, data, `${directory}/${key}`);
        }
    }

    return data;
};


export default function FileTree({
    template,
    treeId,
    viewState,
    onPrimaryAction,
    environment,
}) {
    const data = readTemplate(template);

    return (
        <UncontrolledTreeEnvironment
            dataProvider={new StaticTreeDataProvider(data.items, (item, data) => ({
                ...item,
                data
            }))}
            getItemTitle={item => item.data.key}
            viewState={viewState}
            onPrimaryAction={onPrimaryAction}
        >
            <Tree treeId={treeId} rootItem="root" treeLabel="Tree Example"
                ref={environment} />
        </UncontrolledTreeEnvironment>
    )
}