"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[7392],{8215:function(e,t,a){var n=a(7294);t.Z=function(e){var t=e.children,a=e.hidden,r=e.className;return n.createElement("div",{role:"tabpanel",hidden:a,className:r},t)}},6396:function(e,t,a){a.d(t,{Z:function(){return d}});var n=a(7462),r=a(7294),l=a(2389),o=a(9443);var s=function(){var e=(0,r.useContext)(o.Z);if(null==e)throw new Error('"useUserPreferencesContext" is used outside of "Layout" component.');return e},i=a(3616),u=a(6010),p="tabItem_vU9c";function c(e){var t,a,l,o=e.lazy,c=e.block,d=e.defaultValue,m=e.values,f=e.groupId,v=e.className,g=r.Children.map(e.children,(function(e){if((0,r.isValidElement)(e)&&void 0!==e.props.value)return e;throw new Error("Docusaurus error: Bad <Tabs> child <"+("string"==typeof e.type?e.type:e.type.name)+'>: all children of the <Tabs> component should be <TabItem>, and every <TabItem> should have a unique "value" prop.')})),b=null!=m?m:g.map((function(e){var t=e.props;return{value:t.value,label:t.label,attributes:t.attributes}})),k=(0,i.lx)(b,(function(e,t){return e.value===t.value}));if(k.length>0)throw new Error('Docusaurus error: Duplicate values "'+k.map((function(e){return e.value})).join(", ")+'" found in <Tabs>. Every value needs to be unique.');var h=null===d?d:null!=(t=null!=d?d:null==(a=g.find((function(e){return e.props.default})))?void 0:a.props.value)?t:null==(l=g[0])?void 0:l.props.value;if(null!==h&&!b.some((function(e){return e.value===h})))throw new Error('Docusaurus error: The <Tabs> has a defaultValue "'+h+'" but none of its children has the corresponding value. Available values are: '+b.map((function(e){return e.value})).join(", ")+". If you intend to show no default tab, use defaultValue={null} instead.");var N=s(),y=N.tabGroupChoices,w=N.setTabGroupChoices,D=(0,r.useState)(h),M=D[0],T=D[1],L=[],x=(0,i.o5)().blockElementScrollPositionUntilNextRender;if(null!=f){var O=y[f];null!=O&&O!==M&&b.some((function(e){return e.value===O}))&&T(O)}var _=function(e){var t=e.currentTarget,a=L.indexOf(t),n=b[a].value;n!==M&&(x(t),T(n),null!=f&&w(f,n))},C=function(e){var t,a=null;switch(e.key){case"ArrowRight":var n=L.indexOf(e.currentTarget)+1;a=L[n]||L[0];break;case"ArrowLeft":var r=L.indexOf(e.currentTarget)-1;a=L[r]||L[L.length-1]}null==(t=a)||t.focus()};return r.createElement("div",{className:"tabs-container"},r.createElement("ul",{role:"tablist","aria-orientation":"horizontal",className:(0,u.Z)("tabs",{"tabs--block":c},v)},b.map((function(e){var t=e.value,a=e.label,l=e.attributes;return r.createElement("li",(0,n.Z)({role:"tab",tabIndex:M===t?0:-1,"aria-selected":M===t,key:t,ref:function(e){return L.push(e)},onKeyDown:C,onFocus:_,onClick:_},l,{className:(0,u.Z)("tabs__item",p,null==l?void 0:l.className,{"tabs__item--active":M===t})}),null!=a?a:t)}))),o?(0,r.cloneElement)(g.filter((function(e){return e.props.value===M}))[0],{className:"margin-vert--md"}):r.createElement("div",{className:"margin-vert--md"},g.map((function(e,t){return(0,r.cloneElement)(e,{key:t,hidden:e.props.value!==M})}))))}function d(e){var t=(0,l.Z)();return r.createElement(c,(0,n.Z)({key:String(t)},e))}},9593:function(e,t,a){a.r(t),a.d(t,{contentTitle:function(){return c},default:function(){return v},frontMatter:function(){return p},metadata:function(){return d},toc:function(){return m}});var n=a(7462),r=a(3366),l=(a(7294),a(3905)),o=a(6396),s=a(8215),i=a(9055),u=["components"],p={title:"OpenMLDataset",sidebar_position:1},c="OpenMLDataset",d={unversionedId:"config/adapters/OpenMLDataset",id:"config/adapters/OpenMLDataset",title:"OpenMLDataset",description:"Allows loading a dataset from OpenML.",source:"@site/docs/config/adapters/OpenMLDataset.mdx",sourceDirName:"config/adapters",slug:"/config/adapters/OpenMLDataset",permalink:"/fseval/docs/config/adapters/OpenMLDataset",editUrl:"https://github.com/dunnkers/fseval/tree/website/docs/config/adapters/OpenMLDataset.mdx",tags:[],version:"current",sidebarPosition:1,frontMatter:{title:"OpenMLDataset",sidebar_position:1},sidebar:"tutorialSidebar",previous:{title:"fseval.config.DatasetConfig",permalink:"/fseval/docs/config/DatasetConfig"},next:{title:"WandbDataset",permalink:"/fseval/docs/config/adapters/WandbDataset"}},m=[{value:"Example",id:"example",children:[],level:2}],f={toc:m};function v(e){var t=e.components,a=(0,r.Z)(e,u);return(0,l.kt)("wrapper",(0,n.Z)({},f,a,{components:t,mdxType:"MDXLayout"}),(0,l.kt)("h1",{id:"openmldataset"},"OpenMLDataset"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"class fseval.config.adapters.OpenMLDataset(\n    dataset_id: int=MISSING,\n    target_column: str=MISSING,\n    drop_qualitative: bool=False,\n)\n")),(0,l.kt)("p",null,"Allows loading a dataset from ",(0,l.kt)("a",{parentName:"p",href:"https://www.openml.org/"},"OpenML"),"."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Attributes"),":"),(0,l.kt)("table",null,(0,l.kt)("thead",{parentName:"table"},(0,l.kt)("tr",{parentName:"thead"},(0,l.kt)("th",{parentName:"tr",align:null}),(0,l.kt)("th",{parentName:"tr",align:null}))),(0,l.kt)("tbody",{parentName:"table"},(0,l.kt)("tr",{parentName:"tbody"},(0,l.kt)("td",{parentName:"tr",align:null},"dataset_id : int"),(0,l.kt)("td",{parentName:"tr",align:null},"The dataset ID.")),(0,l.kt)("tr",{parentName:"tbody"},(0,l.kt)("td",{parentName:"tr",align:null},"target_column : str"),(0,l.kt)("td",{parentName:"tr",align:null},"Which column to use as a target. This column will be used as ",(0,l.kt)("inlineCode",{parentName:"td"},"y"),".")),(0,l.kt)("tr",{parentName:"tbody"},(0,l.kt)("td",{parentName:"tr",align:null},"drop_qualitative : bool"),(0,l.kt)("td",{parentName:"tr",align:null},"Whether to drop any column that is not numeric.")),(0,l.kt)("tr",{parentName:"tbody"},(0,l.kt)("td",{parentName:"tr",align:null}),(0,l.kt)("td",{parentName:"tr",align:null})))),(0,l.kt)("h2",{id:"example"},"Example"),(0,l.kt)("p",null,"So, for example, loading the ",(0,l.kt)("a",{parentName:"p",href:"https://www.openml.org/d/61"},"Iris")," dataset:"),(0,l.kt)(o.Z,{groupId:"config-representation",mdxType:"Tabs"},(0,l.kt)(s.Z,{value:"yaml",label:"YAML",default:!0,mdxType:"TabItem"},(0,l.kt)(i.Z,{className:"language-yaml",title:"conf/dataset/iris.yaml",mdxType:"CodeBlock"},"name: Iris Flowers\ntask: classification\nadapter:\n  _target_: fseval.adapters.openml.OpenML\n  dataset_id: 61\n  target_column: class\n")),(0,l.kt)(s.Z,{value:"structured",label:"Structured Config",mdxType:"TabItem"},(0,l.kt)("p",null,"Any dataset can also be configured using Python code. Like so:"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},'from hydra.core.config_store import ConfigStore\nfrom fseval.config import DatasetConfig\nfrom fseval.config.adapters import OpenMLDataset\nfrom fseval.types import Task\n\ncs = ConfigStore.instance()\n\ncs.store(\n    group="dataset",\n    name="iris",\n    node=DatasetConfig(\n        name="Iris Flowers",\n        task=Task.classification,\n        adapter=OpenMLDataset(dataset_id=61, target_column="class"),\n    ),\n)\n')))))}v.isMDXComponent=!0}}]);